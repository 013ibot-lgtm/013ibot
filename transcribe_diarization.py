#!/usr/bin/env python3
from __future__ import annotations
"""
transcribe_diarization.py — faster-whisper + pyannote.audio
                            語音轉文字 + 講者辨識 + 繁體中文輸出

使用方式:
    python transcribe_diarization.py <音訊檔案> [選項]

批次處理:
    python transcribe_diarization.py *.m4a --token <token> --device cuda --model medium
    python transcribe_diarization.py "recordings/*.m4a" --token <token> --skip-existing
    python transcribe_diarization.py "recordings/**/*.m4a" --token <token> --skip-existing

安裝依賴:
    pip install faster-whisper pyannote.audio opencc-python-reimplemented torch soundfile

取得 HuggingFace Token:
    1. 前往 https://huggingface.co 註冊免費帳號
    2. 前往 https://huggingface.co/settings/tokens 取得 Token
    3. 接受以下兩個模型的使用條款：
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import argparse
import bisect
import glob
import hashlib
import json
import time
import subprocess
import datetime
import threading
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

try:
    import torch
except ImportError:
    print("❌ 找不到 torch，請先安裝：pip install torch")
    sys.exit(1)

try:
    from pyannote.audio import Pipeline
except ImportError:
    print("❌ 找不到 pyannote.audio，請先安裝：pip install pyannote.audio")
    sys.exit(1)

try:
    from huggingface_hub import login as hf_login
except ImportError:
    print("❌ 找不到 huggingface_hub，請先安裝：pip install huggingface_hub")
    sys.exit(1)

try:
    import opencc
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False

try:
    import soundfile as sf
except ImportError:
    print("❌ 找不到 soundfile，請先安裝：pip install soundfile")
    sys.exit(1)

import re
import logging
import functools
from contextlib import contextmanager

logger = logging.getLogger(__name__)
MTIME_EPSILON_SEC = 0.01
# 快取結構版本；調整快取資料格式或核心演算法時請遞增。
CACHE_SCHEMA_VERSION = 2
_DIARIZE_GAP_TIERS: tuple[tuple[int | None, float, str], ...] = (
    (180, 0.2, "< 3 分鐘"),
    (900, 0.3, "3–15 分鐘"),
    (3600, 0.5, "15–60 分鐘"),
    (None, 0.8, "> 60 分鐘"),
)
# 分段 diarization 參數
# 超過此長度（秒）自動啟用分段策略，避免把完整 tensor 載入記憶體。
_DIARIZE_CHUNK_THRESHOLD_SEC: float = 600.0   # 10 分鐘
# 每段長度；越短越省記憶體，但段數增加會拉長總處理時間，且邊界越多拼接誤差越多。
_DIARIZE_CHUNK_SEC: float = 300.0             # 5 分鐘
# 前後段重疊；用於比對講者 ID，不計入輸出。應足夠讓兩段都能偵測到同一位講者。
_DIARIZE_OVERLAP_SEC: float = 30.0            # 30 秒
_PUNCT_SP_LEFT_RE = re.compile(r" ([，。！？；：、…])")
_PUNCT_SP_RIGHT_RE = re.compile(r"([，。！？；：、…]) ")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    compute_type: str
    diarize_device: str


@dataclass
class BatchConfig:
    args: argparse.Namespace
    converter: object
    cache_dir: Path
    runtime_cfg: RuntimeConfig


@dataclass
class PipelineState:
    pipeline: Pipeline


def _is_cudnn_symbol_error(err) -> bool:
    e = str(err).lower()
    return (
        "cudnngetlibconfig" in e
        or ("could not load symbol" in e and "cudnn" in e)
    )


def setup_logging(verbose: bool) -> None:
    # 修正 Windows cp950 控制台顯示 emoji 時的 UnicodeEncodeError
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def to_traditional(text: str, converter) -> str:
    return converter.convert(text) if converter else text


def _decompose_seconds(seconds: float) -> tuple[int, int, int, int]:
    sec = max(0.0, float(seconds))
    total = int(sec)
    ms = int((sec - total) * 1000)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return h, m, s, ms


def format_timestamp_srt(seconds: float) -> str:
    h, m, s, ms = _decompose_seconds(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_hms(seconds: float) -> str:
    h, m, s, _ = _decompose_seconds(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detect_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        if device == "cpu" and requested in ("float16", "int8_float16"):
            logger.warning("⚠️  CPU 不支援 %s，已自動改為 int8", requested)
            return "int8"
        if device == "cuda" and requested in ("float16", "int8_float16"):
            try:
                if torch.cuda.get_device_capability()[0] < 7:
                    logger.warning("⚠️  Pascal GPU，%s 不受支援，改為 float32", requested)
                    return "float32"
            except Exception:
                pass
        return requested
    if device == "cpu":
        return "int8"
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7:
            logger.info("ℹ️  GPU CC %d.%d（Pascal）→ float32", *cap)
            return "float32"
        logger.info("ℹ️  GPU CC %d.%d → int8_float16", *cap)
        return "int8_float16"
    except Exception:
        return "float32"


def detect_diarize_device(device: str, override: str = "auto") -> str:
    if override != "auto":
        logger.info("   使用者覆蓋裝置：%s（--diarize-device）", override.upper())
        return override
    if device != "cuda":
        return device
    try:
        if not torch.cuda.is_available():
            logger.warning("   CUDA 不可用，pyannote 改用 CPU")
            return "cpu"
        cap = torch.cuda.get_device_capability()
        logger.info("   自動模式：優先嘗試 CUDA（GPU CC %d.%d）", *cap)
        return "cuda"
    except Exception:
        logger.warning("   無法偵測 CUDA 能力，pyannote 改用 CPU")
        return "cpu"


def check_ffmpeg():
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except FileNotFoundError as e:
        raise RuntimeError("找不到 FFmpeg，請確認已安裝並加入系統 PATH。\n"
                           "下載：https://ffmpeg.org/download.html") from e
    except Exception as e:
        raise RuntimeError(f"檢查 FFmpeg 失敗：{e}") from e
    if r.returncode != 0:
        raise RuntimeError("找不到 FFmpeg，請確認已安裝並加入系統 PATH。\n"
                           "下載：https://ffmpeg.org/download.html")


def _run_ffmpeg_convert(audio_path: str, wav_path: str) -> str:
    logger.info("📂 轉換音訊格式為 WAV（pyannote 需要）...")
    cmd = ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav_path]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg 轉換失敗：{r.stderr}")
    logger.info("✅ 轉換完成：%s", wav_path)
    return wav_path


def _converted_wav_path(audio_path: str) -> str:
    base = audio_path.rsplit(".", 1)[0] + "_converted.wav"
    parent = Path(base).parent
    if os.access(parent, os.W_OK):
        return base
    try:
        key_path = Path(audio_path).resolve()
    except Exception:
        key_path = Path(audio_path).absolute()
    # suffix 由完整路徑雜湊而來，確保不同目錄同名檔案不碰撞
    suffix = hashlib.md5(str(key_path).encode("utf-8")).hexdigest()[:12]
    fallback = Path(tempfile.gettempdir()) / f"{Path(audio_path).stem}_{suffix}_converted.wav"
    logger.warning("⚠️  音訊目錄不可寫入，WAV 暫存改寫至：%s", fallback)
    return str(fallback)


def _is_target_wav(path: str) -> bool:
    """回傳 True 若 path 是 16kHz 單聲道 WAV（內含 sf.info I/O）。"""
    try:
        info = sf.info(path)
    except Exception:
        return False
    return info.format == "WAV" and info.samplerate == 16000 and info.channels == 1


def convert_to_wav(audio_path: str) -> tuple[str, bool]:
    """將音訊轉為 16kHz 單聲道 WAV。

    回傳 (wav_path, newly_created)：
      newly_created=False  → 來源本身符合規格，或復用既有暫存檔；呼叫方不應刪除。
      newly_created=True   → 本次新建暫存檔；呼叫方使用完畢後應刪除。
    """
    if audio_path.lower().endswith(".wav"):
        info = None
        try:
            info = sf.info(audio_path)
        except Exception:
            logger.warning("⚠️  無法讀取 WAV 資訊，將重新轉換...")
        if (
            info is not None
            and info.format == "WAV"
            and info.samplerate == 16000
            and info.channels == 1
        ):
            logger.info("✅ WAV 格式符合規格，略過轉換")
            return audio_path, False          # 來源本身，絕不刪除
        if info is not None and info.format != "WAV":
            logger.warning("⚠️  副檔名為 .wav，但實際格式為 %s，將重新轉換...", info.format)
        elif info is not None and (info.samplerate != 16000 or info.channels != 1):
            logger.warning("⚠️  WAV 格式不符（%d Hz / %d ch），重新轉換...",
                           info.samplerate, info.channels)
    wav_path = _converted_wav_path(audio_path)
    wav_file = Path(wav_path)
    if wav_file.exists():
        try:
            src_mtime = Path(audio_path).stat().st_mtime
            wav_mtime = wav_file.stat().st_mtime
            if wav_mtime + MTIME_EPSILON_SEC >= src_mtime and _is_target_wav(str(wav_file)):
                logger.info("✅ 復用既有 WAV 暫存：%s", wav_file)
                return str(wav_file), False   # 復用既有，保留供下次使用
        except OSError:
            pass
    return _run_ffmpeg_convert(audio_path, wav_path), True  # 新建暫存，用完應刪除


# ──────────────────────────────────────────────
# 講者名稱映射
# ──────────────────────────────────────────────

def build_speaker_map(segments: list, names_str) -> dict:
    first_seen: dict = {}
    for seg in segments:
        for sid in _segment_speaker_ids(seg):
            if sid not in first_seen:
                first_seen[sid] = seg["start"]

    unique_ids = sorted(first_seen, key=lambda sid: (first_seen[sid], sid))
    custom_names = [n.strip() for n in names_str.split(",") if n.strip()] if names_str else []

    speaker_map: dict = {}
    for idx, sid in enumerate(unique_ids):
        speaker_map[sid] = custom_names[idx] if idx < len(custom_names) else f"講者 {idx + 1}"

    if len(custom_names) > len(unique_ids):
        logger.warning(
            "⚠️  --speaker-names 提供 %d 個名字，但僅偵測到 %d 位講者，"
            "多餘的名字將被忽略：%s",
            len(custom_names),
            len(unique_ids),
            "、".join(custom_names[len(unique_ids):]),
        )

    logger.info("\n👤 講者名稱映射（依首次出現時間排序）：")
    for sid, name in speaker_map.items():
        logger.info("   %s（首次出現 %s）  →  %s",
                    sid, format_timestamp_hms(first_seen[sid]), name)
    return speaker_map


def resolve_diarize_gap(requested: str, speaker_segments: list) -> float:
    if requested != "auto":
        try:
            val = float(requested)
        except ValueError:
            logger.warning("⚠️  --diarize-gap 值無效（收到：%s），改用 auto", requested)
        else:
            if val < 0:
                logger.warning(
                    "⚠️  --diarize-gap 必須為非負數（收到：%s），改用 auto",
                    requested,
                )
            else:
                logger.info("ℹ️  diarize-gap：使用者指定 %.2f 秒", val)
                return val

    total_sec = max((s["end"] for s in speaker_segments), default=0.0)
    gap: float | None = None
    tier: str | None = None
    for upper_sec, tier_gap, tier_name in _DIARIZE_GAP_TIERS:
        if upper_sec is None or total_sec < upper_sec:
            gap, tier = tier_gap, tier_name
            break
    if gap is None or tier is None:
        raise RuntimeError("_DIARIZE_GAP_TIERS 缺少 None 終止項")
    logger.info("ℹ️  diarize-gap auto：%.1f 秒音訊 → %.2f 秒（%s）", total_sec, gap, tier)
    return gap


def resolve_speaker_label(speaker_id: str, speaker_map: dict) -> str:
    return speaker_map.get(speaker_id, speaker_id.replace("SPEAKER_", "講者 "))


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _segment_speaker_ids(seg: dict) -> list[str]:
    ids = seg.get("speakers")
    if isinstance(ids, list):
        return _dedupe_keep_order([str(i) for i in ids if i])
    sid = seg.get("speaker")
    return [str(sid)] if sid else []


def resolve_segment_speaker_label(seg: dict, speaker_map: dict) -> str:
    ids = _segment_speaker_ids(seg)
    if not ids:
        return "未知講者"
    labels = [resolve_speaker_label(sid, speaker_map) for sid in ids]
    return " / ".join(_dedupe_keep_order(labels))


def _merge_segment_speaker_ids(left: dict, right: dict) -> list[str]:
    return _dedupe_keep_order(_segment_speaker_ids(left) + _segment_speaker_ids(right))


# ──────────────────────────────────────────────
# 核心功能
# ──────────────────────────────────────────────

def load_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    logger.info("\n🤖 載入 faster-whisper 模型（%s / %s / %s）...", model_size, device, compute_type)
    logger.info("   首次使用會自動下載模型，請稍候...\n")
    try:
        from faster_whisper import WhisperModel as WhisperModelRuntime
    except ImportError as e:
        raise RuntimeError("找不到 faster-whisper，請先安裝：pip install faster-whisper") from e
    # 延遲匯入可避免 Windows CUDA DLL 載入順序問題。
    return WhisperModelRuntime(model_size, device=device, compute_type=compute_type)


def run_transcription(model: WhisperModel, audio_path: str,
                      language: str, beam_size: int, word_timestamps: bool) -> list:
    logger.info("🎙️  開始語音辨識：%s", audio_path)
    lang = None if language == "auto" else language
    segments_gen, info = model.transcribe(
        audio_path, language=lang, beam_size=beam_size,
        word_timestamps=word_timestamps, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
    )
    logger.info("📊 語言：%s（%.1f%%）  音訊長度：%.1f 秒\n",
                info.language, info.language_probability * 100, info.duration)
    segs = list(segments_gen)
    logger.info("✅ 語音辨識完成！共 %d 個片段", len(segs))
    return segs


# ──────────────────────────────────────────────
# torch.load patch 加互斥鎖（防止並行載入時的競態條件）
# ──────────────────────────────────────────────
_torch_load_patch_lock = threading.Lock()


@contextmanager
def _patch_torch_load_for_pyannote():
    """以互斥鎖保護 torch.load 全域 patch，防止並行載入時的競態條件。"""
    with _torch_load_patch_lock:
        orig_ser  = torch.serialization.load
        orig_load = torch.load

        @functools.wraps(orig_ser)
        def _patched(f, *args, **kwargs):
            kwargs["weights_only"] = False
            return orig_ser(f, *args, **kwargs)

        try:
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        except Exception:
            pass
        torch.serialization.load = _patched
        torch.load               = _patched
        try:
            yield
        finally:
            torch.serialization.load = orig_ser
            torch.load               = orig_load


def load_diarization_pipeline(token: str, device: str,
                               diarize_device_override: str = "auto") -> tuple:
    logger.info("\n👥 載入 pyannote 講者辨識模型...")
    hf_login(token=token, add_to_git_credential=False)
    with _patch_torch_load_for_pyannote():
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarize_device = detect_diarize_device(device, override=diarize_device_override)
    # 保留 try/except fallback 作為第二道防線（即使 detect_diarize_device 已判斷）
    try:
        pipeline = pipeline.to(torch.device(diarize_device))
    except Exception as e:
        if diarize_device != "cpu":
            logger.warning("   pyannote 切到 %s 失敗（%s），自動回退 CPU",
                           diarize_device.upper(), e)
            pipeline = pipeline.to(torch.device("cpu"))
            diarize_device = "cpu"
        else:
            raise
    logger.info("✅ pyannote 載入完成，裝置：%s", diarize_device.upper())
    return pipeline, diarize_device


def _read_wav_chunk(wav_path: str, start_sec: float, chunk_sec: float,
                    sample_rate: int) -> torch.Tensor:
    """讀取 WAV 指定片段，回傳 (channels, frames) tensor，不載入整個檔案。"""
    start_frame = int(start_sec * sample_rate)
    n_frames    = int(chunk_sec  * sample_rate)
    with sf.SoundFile(wav_path) as f:
        f.seek(start_frame)
        data = f.read(n_frames, dtype="float32", always_2d=True)
    return torch.tensor(data.T)


def _remap_speakers_by_overlap(
    prev_segs: list[dict],
    cur_segs:  list[dict],
    overlap_start: float,
    overlap_end:   float,
    next_speaker_idx: int,
) -> tuple[list[dict], int]:
    """在重疊區間內比對前後兩段的講者 ID，回傳 (重新映射後的 cur_segs, 更新後的全域計數器)。

    演算法：
    1. 計算 prev / cur 兩側各講者在重疊區間內的累積時長。
    2. 以交集時長矩陣做貪婪最佳匹配（最長交集優先，每個 prev 講者最多被用一次）。
    3. 無法匹配的 cur 講者配發新的全域 ID（SPEAKER_XX）。
    """
    def _coverage(segs: list[dict], t0: float, t1: float) -> dict[str, float]:
        cov: dict[str, float] = {}
        for s in segs:
            lo = max(s["start"], t0)
            hi = min(s["end"],   t1)
            if hi > lo:
                cov[s["speaker"]] = cov.get(s["speaker"], 0.0) + (hi - lo)
        return cov

    prev_cov = _coverage(prev_segs, overlap_start, overlap_end)
    cur_cov  = _coverage(cur_segs,  overlap_start, overlap_end)
    remap: dict[str, str] = {}

    if prev_cov and cur_cov:
        # 建立 (cur_id, prev_id) → 交集時長的矩陣
        cross: dict[tuple[str, str], float] = {}
        for s_c in cur_segs:
            for s_p in prev_segs:
                lo = max(s_c["start"], s_p["start"], overlap_start)
                hi = min(s_c["end"],   s_p["end"],   overlap_end)
                if hi > lo:
                    key = (s_c["speaker"], s_p["speaker"])
                    cross[key] = cross.get(key, 0.0) + (hi - lo)

        # 貪婪匹配：交集時長降序，每個 prev ID 最多用一次
        used_prev: set[str] = set()
        for (cur_id, prev_id), _ in sorted(cross.items(), key=lambda x: -x[1]):
            if cur_id not in remap and prev_id not in used_prev:
                remap[cur_id] = prev_id
                used_prev.add(prev_id)

    # 未匹配的 cur 講者→配新 ID
    for cur_id in cur_cov:
        if cur_id not in remap:
            remap[cur_id] = f"SPEAKER_{next_speaker_idx:02d}"
            next_speaker_idx += 1

    remapped = [{**s, "speaker": remap.get(s["speaker"], s["speaker"])}
                for s in cur_segs]
    return remapped, next_speaker_idx


def _diarize_chunked(
    wav_path: str,
    pipeline_fn,
    chunk_sec:   float = _DIARIZE_CHUNK_SEC,
    overlap_sec: float = _DIARIZE_OVERLAP_SEC,
) -> list[dict]:
    """真正的分段 diarization：每次只載入 chunk_sec 秒音訊，以重疊區間拼接講者 ID。

    記憶體峰值 ≈ 單段大小（chunk_sec × sample_rate × 4 bytes），
    與整段載入相比可大幅降低 RAM / VRAM 用量。
    代價：邊界附近的講者切換準確度略低於全段推論；overlap_sec 越大邊界越穩。
    """
    if chunk_sec <= overlap_sec:
        raise ValueError(
            f"chunk_sec ({chunk_sec}) 必須大於 overlap_sec ({overlap_sec})"
        )

    info       = sf.info(wav_path)
    total_sec  = info.duration
    sample_rate = info.samplerate
    step_sec   = chunk_sec - overlap_sec

    # 計算各段起點（最後一段延伸到 total_sec）
    chunk_starts: list[float] = []
    t = 0.0
    while t < total_sec:
        chunk_starts.append(t)
        t += step_sec
    n_chunks = len(chunk_starts)

    logger.info(
        "   分段 diarization：%.1f 秒音訊 → %d 段"
        "（每段 %.0f 秒 / 重疊 %.0f 秒）",
        total_sec, n_chunks, chunk_sec, overlap_sec,
    )

    all_segs:          list[dict] = []
    prev_segs_global:  list[dict] = []   # 前段 segments（全域時間），供重疊比對
    global_spk_idx:    int        = 0

    for i, t_start in enumerate(chunk_starts):
        t_end   = min(t_start + chunk_sec, total_sec)
        dur     = t_end - t_start

        logger.info("   段 %d/%d：%.1f–%.1f 秒...", i + 1, n_chunks, t_start, t_end)

        wt = _read_wav_chunk(wav_path, t_start, dur, sample_rate)
        try:
            result = pipeline_fn({"waveform": wt, "sample_rate": sample_rate})
        finally:
            del wt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ann = getattr(result, "speaker_diarization", result)
        if not hasattr(ann, "itertracks"):
            raise RuntimeError(f"分段 {i+1} 無法解析 DiarizeOutput：{dir(result)}")

        # 本段 segments 轉換到全域時間
        chunk_segs: list[dict] = [
            {"start": trn.start + t_start, "end": trn.end + t_start, "speaker": spk}
            for trn, _, spk in ann.itertracks(yield_label=True)
        ]

        if i == 0:
            # 第一段：直接配全域 ID（SPEAKER_00, 01, ...）
            local_ids = sorted({s["speaker"] for s in chunk_segs})
            id_map = {lid: f"SPEAKER_{global_spk_idx + j:02d}"
                      for j, lid in enumerate(local_ids)}
            global_spk_idx += len(local_ids)
            chunk_segs = [{**s, "speaker": id_map[s["speaker"]]} for s in chunk_segs]
            all_segs.extend(chunk_segs)
        else:
            overlap_start_g = t_start
            overlap_end_g   = min(t_start + overlap_sec, t_end)

            # 重疊區間比對，更新 cur_segs 的講者 ID
            chunk_segs, global_spk_idx = _remap_speakers_by_overlap(
                prev_segs_global, chunk_segs,
                overlap_start_g, overlap_end_g,
                global_spk_idx,
            )

            # 只保留「貢獻區間」（重疊後半段起），裁切跨越邊界的 segment
            contribution_start = overlap_end_g
            contrib: list[dict] = []
            for s in chunk_segs:
                if s["end"] <= contribution_start:
                    continue                          # 完全在重疊區，跳過
                clipped_start = max(s["start"], contribution_start)
                contrib.append({**s, "start": clipped_start})
            all_segs.extend(contrib)

        prev_segs_global = chunk_segs

    result_segs = sorted(all_segs, key=lambda s: s["start"])
    unique = {s["speaker"] for s in result_segs}
    logger.info(
        "   分段拼接完成：%d 個 segments，%d 位講者（%s）",
        len(result_segs), len(unique), ", ".join(sorted(unique)),
    )
    return result_segs


def run_diarization_with_pipeline(wav_path: str, pipeline, num_speakers) -> list:
    logger.info("\n🔍 開始分析講者（此步驟需要數分鐘）...")
    params = {"num_speakers": num_speakers} if num_speakers else {}
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError:
        ProgressHook = None

    def _run_with_optional_hook(audio_input):
        if ProgressHook is None:
            return pipeline(audio_input, **params)
        try:
            with ProgressHook() as hook:
                return pipeline(audio_input, hook=hook, **params)
        except Exception:
            return pipeline(audio_input, **params)

    # speaker_segments: 由分段路徑直接產生（list[dict]）
    # diarization:      由全段路徑產生（annotation 物件），待後處理提取
    speaker_segments: list[dict] | None = None
    diarization = None

    try:
        # ── 策略 1：傳路徑（最快，讓 pyannote 自行處理 I/O）
        try:
            audio_input = {"uri": Path(wav_path).stem, "audio": wav_path}
            logger.info("   📌 策略 1：傳檔案路徑")
            diarization = _run_with_optional_hook(audio_input)
            logger.info("   ✅ 策略 1 成功")
        except Exception as e1:
            if diarization is not None:
                logger.warning(
                    "   ⚠️  策略 1 後處理失敗（%s），但 diarization 已完成，沿用既有結果",
                    e1,
                )
            else:
                logger.warning("   ⚠️  策略 1 失敗（%s），嘗試後續策略...", e1)

        # ── 策略 2：真正的分段 diarization（長音檔專用，降低記憶體峰值）
        # 每次只載入 _DIARIZE_CHUNK_SEC 秒音訊，以重疊區間比對講者 ID 後拼接。
        # 記憶體峰值 ≈ 單段大小，適合數小時錄音；邊界準確度略低於全段推論。
        if diarization is None and speaker_segments is None:
            try:
                total_sec = sf.info(wav_path).duration
                if total_sec >= _DIARIZE_CHUNK_THRESHOLD_SEC:
                    logger.info(
                        "   📌 策略 2：分段 diarization"
                        "（音檔 %.1f 秒 ≥ 門檻 %.0f 秒，每段 %.0f 秒 / 重疊 %.0f 秒）",
                        total_sec, _DIARIZE_CHUNK_THRESHOLD_SEC,
                        _DIARIZE_CHUNK_SEC, _DIARIZE_OVERLAP_SEC,
                    )
                    speaker_segments = _diarize_chunked(
                        wav_path,
                        pipeline_fn=_run_with_optional_hook,
                    )
                    logger.info("   ✅ 策略 2 成功")
                else:
                    logger.info(
                        "   ℹ️  音檔 %.1f 秒 < 門檻 %.0f 秒，略過分段策略",
                        total_sec, _DIARIZE_CHUNK_THRESHOLD_SEC,
                    )
            except Exception as e2:
                logger.warning("   ⚠️  策略 2 失敗（%s），回退全段載入...", e2)
                speaker_segments = None

        # ── 策略 3：全段讀檔後合併為完整 tensor（繞過部分環境 sf.read 相容性問題）
        # 注意：最終仍以 torch.cat 合併為單一 tensor，記憶體峰值與策略 4 相當，
        # cat 過程中甚至短暫更高。長音檔建議優先讓策略 2 處理。
        if diarization is None and speaker_segments is None:
            try:
                logger.info(
                    "   📌 策略 3：分段讀檔後合併 tensor"
                    "（非 streaming，記憶體用量與策略 4 相當）"
                )
                chunks = []
                with sf.SoundFile(wav_path) as f:
                    sample_rate = f.samplerate
                    while True:
                        block = f.read(16000 * 60, dtype="float32", always_2d=True)
                        if not len(block):
                            break
                        chunks.append(torch.tensor(block.T))
                wt = torch.cat(chunks, dim=1); del chunks
                logger.warning(
                    "   ⚠️  音檔完整 tensor 已載入記憶體（%.1f 秒），"
                    "超長音檔可能造成 RAM / VRAM 不足",
                    wt.shape[1] / sample_rate,
                )
                try:
                    audio_input = {"waveform": wt, "sample_rate": sample_rate}
                    diarization = _run_with_optional_hook(audio_input)
                finally:
                    del wt
                logger.info("   ✅ 策略 3 成功")
            except Exception as e3:
                logger.warning("   ⚠️  策略 3 失敗（%s），回退一次性讀取...", e3)

        # ── 策略 4：一次性讀取（最後防線）
        if diarization is None and speaker_segments is None:
            logger.info("   📌 策略 4：一次性讀取")
            waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
            wt = torch.tensor(waveform.T)
            audio_input = {"waveform": wt, "sample_rate": sample_rate}
            try:
                diarization = _run_with_optional_hook(audio_input)
            finally:
                del wt

        # ── 全段路徑：從 annotation 物件提取 segments
        if speaker_segments is None:
            annotation = (diarization.speaker_diarization
                          if hasattr(diarization, "speaker_diarization")
                          else diarization)
            if not hasattr(annotation, "itertracks"):
                raise RuntimeError(f"無法解析 DiarizeOutput，屬性：{dir(diarization)}")
            speaker_segments = [
                {"start": turn.start, "end": turn.end, "speaker": speaker}
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]

        unique = {s["speaker"] for s in speaker_segments}
        logger.info(
            "✅ 講者辨識完成！偵測到 %d 位：%s",
            len(unique), ", ".join(sorted(unique)),
        )
        return speaker_segments
    finally:
        # 統一在每次 diarization 後釋放可回收的 CUDA 快取，降低批次處理碎片化。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────
# 後處理
# ──────────────────────────────────────────────

def _merge_adjacent_speaker_segments(speaker_segments: list, gap_sec: float) -> list:
    if not speaker_segments:
        return speaker_segments
    segs = sorted(speaker_segments, key=lambda s: s["start"])
    merged = [dict(segs[0])]
    for seg in segs[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] <= gap_sec:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


def _collapse_short_speaker_islands(
    speaker_segments: list,
    gap_sec: float,
    island_max_sec: float = 0.6,
) -> tuple[list, int]:
    if len(speaker_segments) < 3:
        return speaker_segments, 0

    relabeled = [dict(seg) for seg in speaker_segments]
    collapsed = 0
    for i in range(1, len(relabeled) - 1):
        prev_seg = relabeled[i - 1]
        cur_seg = relabeled[i]
        next_seg = relabeled[i + 1]
        if prev_seg["speaker"] != next_seg["speaker"]:
            continue
        if cur_seg["speaker"] == prev_seg["speaker"]:
            continue

        island_dur = cur_seg["end"] - cur_seg["start"]
        gap_left = cur_seg["start"] - prev_seg["end"]
        gap_right = next_seg["start"] - cur_seg["end"]
        if island_dur <= island_max_sec and gap_left <= gap_sec and gap_right <= gap_sec:
            cur_seg["speaker"] = prev_seg["speaker"]
            collapsed += 1

    return _merge_adjacent_speaker_segments(relabeled, gap_sec=gap_sec), collapsed


def smooth_speaker_segments(speaker_segments: list, gap_sec: float = 0.3) -> list:
    merged = _merge_adjacent_speaker_segments(speaker_segments, gap_sec=gap_sec)
    total_collapsed = 0
    max_rounds = max(1, len(merged) * 2)
    rounds = 0
    while rounds < max_rounds:
        merged, collapsed = _collapse_short_speaker_islands(merged, gap_sec=gap_sec)
        if collapsed == 0:
            break
        total_collapsed += collapsed
        rounds += 1
    if rounds >= max_rounds:
        logger.warning("⚠️  孤島修正達到上限（%d 輪），已提前停止", max_rounds)
    if total_collapsed:
        logger.debug("   孤島修正：合併 %d 個極短異講者片段", total_collapsed)
    return merged


def _is_cjk(text: str) -> bool:
    n = len(text)
    if n == 0:
        return False
    threshold = n * 0.3
    cjk = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3040 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7A3
        ):
            cjk += 1
            if cjk > threshold:
                return True
    return False


def _join_text(a: str, b: str) -> str:
    a, b = a.strip(), b.strip()
    if not b: return a
    if not a: return b
    return a + ("" if _is_cjk(a) else " ") + b


def _is_sentence_end(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in set("。！？…!?")


_OVERLAP_RATIO_MIN = 0.05


def _resolve_speaker(
    w_start,
    w_end,
    s_segs,
    s_mids,
    s_mid_times,
    sp_ptr,
    n_sp,
    overlap_labels: bool = False,
    overlap_ratio_min: float = 0.35,
):
    w_dur = max(w_end - w_start, 1e-6)
    while sp_ptr < n_sp and s_segs[sp_ptr]["end"] <= w_start:
        sp_ptr += 1
    best_speaker, best_ratio = None, 0.0
    overlap_candidates: list[tuple[float, str]] = []
    j = sp_ptr
    while j < n_sp and s_segs[j]["start"] < w_end:
        sp    = s_segs[j]
        ratio = (min(w_end, sp["end"]) - max(w_start, sp["start"])) / w_dur
        if ratio > best_ratio:
            best_ratio, best_speaker = ratio, sp["speaker"]
        if overlap_labels and ratio >= overlap_ratio_min:
            overlap_candidates.append((ratio, sp["speaker"]))
        j += 1
    if best_speaker is None or best_ratio < _OVERLAP_RATIO_MIN:
        # 重疊不足時改用最近中點講者。此分支只覆寫主講者 best_speaker，
        # 不會清空 overlap_candidates；後續仍會把候選清單合併進 speaker_ids。
        w_mid = (w_start + w_end) / 2
        pos = bisect.bisect_left(s_mid_times, w_mid)
        if pos <= 0:
            best_speaker = s_mids[0][0]["speaker"]
        elif pos >= n_sp:
            best_speaker = s_mids[-1][0]["speaker"]
        else:
            left_sp, left_mid = s_mids[pos - 1]
            right_sp, right_mid = s_mids[pos]
            if abs(w_mid - left_mid) <= abs(right_mid - w_mid):
                best_speaker = left_sp["speaker"]
            else:
                best_speaker = right_sp["speaker"]
    speaker_ids = [best_speaker]
    if overlap_labels and overlap_candidates:
        overlap_candidates.sort(key=lambda x: x[0], reverse=True)
        speaker_ids = _dedupe_keep_order(
            [best_speaker] + [sid for _, sid in overlap_candidates]
        )
    return best_speaker, speaker_ids, sp_ptr


def _prepare_speaker_lookup(smoothed: list) -> tuple[list, list, list, int]:
    s_segs = smoothed
    # s_segs 必須按 start 升序（_resolve_speaker 的 sp_ptr 單向掃描依賴此不變式）。
    if __debug__ and any(s_segs[i]["start"] > s_segs[i + 1]["start"] for i in range(len(s_segs) - 1)):
        raise AssertionError("internal error: s_segs must be sorted by start")
    s_mids = [(sp, (sp["start"] + sp["end"]) / 2) for sp in s_segs]
    s_mid_times = [mid for _, mid in s_mids]
    n_sp = len(s_segs)
    return s_segs, s_mids, s_mid_times, n_sp


def assign_speakers(whisper_segments: list, speaker_segments: list,
                    gap_sec: float = 0.3,
                    overlap_labels: bool = False,
                    overlap_ratio_min: float = 0.35) -> list:
    if not speaker_segments:
        return [{"start": seg.start, "end": seg.end,
                 "speaker": "未知講者", "speakers": ["未知講者"], "text": seg.text.strip()}
                for seg in whisper_segments]
    smoothed = smooth_speaker_segments(speaker_segments, gap_sec=gap_sec)
    s_segs, s_mids, s_mid_times, n_sp = _prepare_speaker_lookup(smoothed)
    result = []
    sp_ptr = 0
    for seg in sorted(whisper_segments, key=lambda s: s.start):
        w_start, w_end = seg.start, seg.end
        best_speaker, speaker_ids, sp_ptr = _resolve_speaker(
            w_start, w_end, s_segs, s_mids, s_mid_times, sp_ptr, n_sp,
            overlap_labels=overlap_labels, overlap_ratio_min=overlap_ratio_min,
        )
        result.append({"start": seg.start, "end": seg.end,
                        "speaker": best_speaker, "speakers": speaker_ids, "text": seg.text.strip()})
    return result


def _get_word_items(seg) -> list:
    words = getattr(seg, "words", None)
    if words is None:
        # 未啟用字詞時間戳
        return []
    if not words:
        # 啟用字詞時間戳，但此片段沒有可用詞（例如靜音段）
        return []
    return [
        {"start": w["start"], "end": w["end"], "word": w["word"]}
        if isinstance(w, dict)
        else {"start": w.start, "end": w.end, "word": w.word}
        for w in words
    ]


def assign_speakers_word_level(whisper_segments, speaker_segments,
                                gap_sec=0.3, sentence_split_gap=0.8,
                                hard_split_gap=2.5,
                                overlap_labels: bool = False,
                                overlap_ratio_min: float = 0.35):
    flat_words = []
    has_any_words = False
    for seg in whisper_segments:
        words = _get_word_items(seg)
        if words:
            has_any_words = True
            flat_words.extend(words)
        else:
            flat_words.append({"start": seg.start, "end": seg.end, "word": seg.text.strip()})

    if not has_any_words:
        logger.warning(
            "⚠️  segments 均無 word timestamps，降回 segment 級對齊。\n"
            "   （確認已啟用字詞時間戳；若要關閉請加 --no-word-timestamps）"
        )
        return assign_speakers(
            whisper_segments, speaker_segments, gap_sec=gap_sec,
            overlap_labels=overlap_labels, overlap_ratio_min=overlap_ratio_min,
        )

    if not flat_words:
        return []

    logger.info("   字詞級對齊：共 %d 個字詞", len(flat_words))

    if not speaker_segments:
        return [{"start": w["start"], "end": w["end"],
                 "speaker": "未知講者", "speakers": ["未知講者"], "text": w["word"]}
                for w in flat_words]

    smoothed = smooth_speaker_segments(speaker_segments, gap_sec=gap_sec)
    s_segs, s_mids, s_mid_times, n_sp = _prepare_speaker_lookup(smoothed)

    labeled = []
    sp_ptr = 0
    for w in flat_words:
        best_speaker, speaker_ids, sp_ptr = _resolve_speaker(
            w["start"], w["end"], s_segs, s_mids, s_mid_times, sp_ptr, n_sp,
            overlap_labels=overlap_labels, overlap_ratio_min=overlap_ratio_min,
        )
        labeled.append({**w, "speaker": best_speaker, "speakers": speaker_ids})

    proto_segments = []
    cur = {
        "start":   labeled[0]["start"],
        "end":     labeled[0]["end"],
        "speaker": labeled[0]["speaker"],
        "speakers": labeled[0]["speakers"],
        "text":    labeled[0]["word"],
    }
    speaker_switches = 0
    pause_splits     = 0   # c2：同講者強制靜音切段
    sentence_splits  = 0   # c3：同講者句末停頓切段

    for lw in labeled[1:]:
        gap          = lw["start"] - cur["end"]
        same_speaker = (lw["speaker"] == cur["speaker"])
        c1 = not same_speaker
        c2 = same_speaker and gap >= hard_split_gap
        c3 = same_speaker and gap >= sentence_split_gap and _is_sentence_end(cur["text"])
        if c1 or c2 or c3:
            proto_segments.append(cur)
            if c1:
                speaker_switches += 1
            elif c2:
                pause_splits += 1
            elif c3:
                sentence_splits += 1
            cur = {
                "start":   lw["start"],
                "end":     lw["end"],
                "speaker": lw["speaker"],
                "speakers": lw["speakers"],
                "text":    lw["word"],
            }
        else:
            cur["end"]  = lw["end"]
            cur["text"] = _join_text(cur["text"], lw["word"])
            cur["speakers"] = _dedupe_keep_order(cur["speakers"] + lw["speakers"])

    proto_segments.append(cur)
    logger.info(
        "   字詞級對齊：%d 字詞 → %d proto-segments"
        "（講者切換 %d 次 / 強制靜音切段 %d 次 / 句末停頓切段 %d 次）",
        len(labeled), len(proto_segments),
        speaker_switches, pause_splits, sentence_splits,
    )
    return proto_segments


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)
    text = _PUNCT_SP_LEFT_RE.sub(r"\1", text)
    text = _PUNCT_SP_RIGHT_RE.sub(r"\1", text)
    text = re.sub(r"([。！？…!?])\1+", r"\1", text)
    return text


def merge_consecutive(segments, gap_hard_sec=3.0,
                      gap_mid_sec=0.5, max_duration_sec=60.0,
                      max_duration_hard_sec=120.0):
    """合併連續片段。

    判斷順序先看片段長度上限，再看停頓門檻：也就是當合併後長度尚未超過
    `max_duration_hard_sec` 時，可能即使 `gap >= gap_hard_sec` 仍會被合併。
    """
    if not segments:
        return segments
    if max_duration_hard_sec <= max_duration_sec:
        logger.warning("⚠️  max_duration_hard_sec 小於或等於 max_duration_sec，自動調整為 2x")
        max_duration_hard_sec = max_duration_sec * 2
        logger.warning("   已自動調整為 %.1f 秒", max_duration_hard_sec)

    first = dict(segments[0])
    first["text"] = _normalize_text(first["text"])
    merged = [first]

    for seg in segments[1:]:
        last     = merged[-1]
        gap      = seg["start"] - last["end"]
        same_spk = seg["speaker"] == last["speaker"]
        # 預估「合併後」片段長度，用於限制段落過長。
        merged_dur = seg["end"] - last["start"]
        seg_text = _normalize_text(seg["text"])

        if not same_spk:
            merged.append({**seg, "text": seg_text}); continue
        if merged_dur > max_duration_hard_sec:
            merged.append({**seg, "text": seg_text}); continue
        if merged_dur > max_duration_sec:
            if _is_sentence_end(last["text"]):
                merged.append({**seg, "text": seg_text})
            else:
                last["end"]  = seg["end"]
                last["text"] = _join_text(last["text"], seg_text)
                last["speakers"] = _merge_segment_speaker_ids(last, seg)
            continue
        if gap >= gap_hard_sec:
            merged.append({**seg, "text": seg_text}); continue
        if gap < gap_mid_sec:
            last["end"]  = seg["end"]
            last["text"] = _join_text(last["text"], seg_text)
            last["speakers"] = _merge_segment_speaker_ids(last, seg)
            continue
        if _is_sentence_end(last["text"]):
            merged.append({**seg, "text": seg_text})
        else:
            last["end"]  = seg["end"]
            last["text"] = _join_text(last["text"], seg_text)
            last["speakers"] = _merge_segment_speaker_ids(last, seg)

    return merged


def _absorb_short_interjections_one_pass(
    segments: list,
    min_chars: int = 4,
    same_speaker_only: bool = True,
    max_gap_sec: float = 0.8,
) -> tuple[list, int]:
    if not segments:
        return segments, 0

    if max_gap_sec < 0:
        logger.warning("⚠️ max_gap_sec 小於 0，已自動視為 0")
        max_gap_sec = 0.0

    # 使用 dict 複製，避免修改時回寫到傳入參數的原始物件。
    result = [dict(seg) for seg in segments]
    absorbed = [False] * len(result)
    frozen = [False] * len(result)
    n = len(result)

    for i, seg in enumerate(result):
        if absorbed[i] or frozen[i]:
            continue

        seg_text = _normalize_text(seg["text"])
        if len(seg_text) > min_chars:
            continue

        # frozen 片段代表剛吸收過鄰近短句，避免在同一輪被再次當作吸收目標。
        has_prev = i > 0 and not absorbed[i - 1] and not frozen[i - 1]
        has_next = i < n - 1 and not absorbed[i + 1] and not frozen[i + 1]

        prev_text = _normalize_text(result[i - 1]["text"]) if has_prev else ""
        next_text = _normalize_text(result[i + 1]["text"]) if has_next else ""

        gap_prev = seg["start"] - result[i - 1]["end"] if has_prev else float("inf")
        gap_next = result[i + 1]["start"] - seg["end"] if has_next else float("inf")

        prev_ok = (
            has_prev
            and gap_prev <= max_gap_sec
            and not _is_sentence_end(prev_text)
            and (not same_speaker_only or seg["speaker"] == result[i - 1]["speaker"])
        )
        next_ok = (
            has_next
            and gap_next <= max_gap_sec
            and not _is_sentence_end(seg_text)
            and (not same_speaker_only or seg["speaker"] == result[i + 1]["speaker"])
        )

        if prev_ok and next_ok:
            absorb_to_prev = gap_prev <= gap_next
        else:
            absorb_to_prev = prev_ok

        if absorb_to_prev:
            prev = result[i - 1]
            prev["end"] = seg["end"]
            prev["text"] = _join_text(prev_text, seg_text)
            prev["speakers"] = _merge_segment_speaker_ids(prev, seg)
            absorbed[i] = True
            logger.debug(
                " 吸收超短插話進前段（gap=%.2fs）：「%s」→ 【%s】",
                gap_prev, seg_text, result[i - 1]["speaker"]
            )
        elif next_ok:
            nxt = result[i + 1]
            nxt["start"] = seg["start"]
            nxt["text"] = _join_text(seg_text, next_text)
            nxt["speakers"] = _merge_segment_speaker_ids(seg, nxt)
            absorbed[i] = True
            frozen[i + 1] = True
            logger.debug(
                " 吸收超短插話進後段（gap=%.2fs）：「%s」→ 【%s】",
                gap_next, seg_text, result[i + 1]["speaker"]
            )
        else:
            logger.debug(
                " 保留超短插話（gap_prev=%.2fs, gap_next=%.2fs, max_gap_sec=%.2fs）：「%s」",
                gap_prev if has_prev else -1.0,
                gap_next if has_next else -1.0,
                max_gap_sec,
                seg_text,
            )

    kept = [seg for i, seg in enumerate(result) if not absorbed[i]]
    absorbed_count = sum(absorbed)
    return kept, absorbed_count

def absorb_short_interjections(
    segments: list,
    min_chars: int = 4,
    same_speaker_only: bool = True,
    max_gap_sec: float = 0.8,
) -> list:
    """把超短插話吸收進前後文，避免碎片行。"""
    if not segments:
        return segments

    current = segments
    total_absorbed = 0
    rounds = 0
    max_rounds = max(1, len(current) * 2)

    while rounds < max_rounds:
        current, absorbed_count = _absorb_short_interjections_one_pass(
            current,
            min_chars=min_chars,
            same_speaker_only=same_speaker_only,
            max_gap_sec=max_gap_sec,
        )
        if absorbed_count == 0:
            break
        total_absorbed += absorbed_count
        rounds += 1

    if rounds >= max_rounds:
        logger.warning("⚠️ 超短插話吸收達到上限（%d 輪），已提前停止", max_rounds)

    if total_absorbed:
        logger.info(
            " 吸收超短插話：%d 個（≤ %d 字，%d 輪，max_gap_sec=%.2f）",
            total_absorbed, min_chars, rounds, max_gap_sec
        )

    return current

# ──────────────────────────────────────────────
# 輸出函式
# ──────────────────────────────────────────────

def _atomic_write_text(output_path: Path, writer) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f"{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            writer(f)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def save_txt(segments: list, output_path: Path, converter,
             audio_file: str, speaker_map: dict, show_timestamps: bool = True,
             title: str = "會議逐字稿"):
    def _write(f):
        f.write("=" * 60 + f"\n  {title}\n")
        f.write(f"  音訊檔案：{audio_file}\n")
        f.write(f"  產生時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        current_speaker_ids: tuple[str, ...] | None = None
        for seg in segments:
            label = resolve_segment_speaker_label(seg, speaker_map)
            text  = to_traditional(seg["text"], converter)
            seg_ids = tuple(_segment_speaker_ids(seg))
            if seg_ids != current_speaker_ids:
                f.write(f"\n【{label}】\n")
                current_speaker_ids = seg_ids
            if show_timestamps:
                ts = f"[{format_timestamp_hms(seg['start'])} → {format_timestamp_hms(seg['end'])}]"
                f.write(f"  {ts} {text}\n")
            else:
                f.write(f"  {text}\n")
    _atomic_write_text(output_path, _write)
    logger.info("✅ 純文字逐字稿已儲存：%s", output_path)


def save_srt(segments: list, output_path: Path, converter, speaker_map: dict):
    def _write(f):
        for i, seg in enumerate(segments, 1):
            label = resolve_segment_speaker_label(seg, speaker_map)
            text  = to_traditional(seg["text"], converter)
            f.write(f"{i}\n{format_timestamp_srt(seg['start'])} --> "
                    f"{format_timestamp_srt(seg['end'])}\n[{label}] {text}\n\n")
    _atomic_write_text(output_path, _write)
    logger.info("✅ SRT 字幕已儲存：%s", output_path)


def save_vtt(segments: list, output_path: Path, converter, speaker_map: dict):
    def _write(f):
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            label = resolve_segment_speaker_label(seg, speaker_map)
            text  = to_traditional(seg["text"], converter)
            s = format_timestamp_srt(seg["start"]).replace(",", ".")
            e = format_timestamp_srt(seg["end"]).replace(",", ".")
            f.write(f"{i}\n{s} --> {e}\n[{label}] {text}\n\n")
    _atomic_write_text(output_path, _write)
    logger.info("✅ VTT 字幕已儲存：%s", output_path)


def save_json(segments: list, output_path: Path, converter,
              audio_file: str, speaker_map: dict):
    meta = {
        "audio_file": audio_file,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    speakers = [{"id": sid, "name": name} for sid, name in speaker_map.items()]

    def _write(f):
        f.write("{\n")
        f.write('  "version": 1,\n')
        f.write('  "meta": ')
        json.dump(meta, f, ensure_ascii=False)
        f.write(",\n")
        f.write('  "speakers": ')
        json.dump(speakers, f, ensure_ascii=False)
        f.write(",\n")
        f.write('  "segments": [\n')
        for i, seg in enumerate(segments, 1):
            seg_speaker_ids = _segment_speaker_ids(seg)
            item = {
                "index": i,
                "speaker_id": seg["speaker"],
                "speaker": resolve_speaker_label(seg["speaker"], speaker_map),
                "speaker_ids": seg_speaker_ids,
                "speakers": [
                    resolve_speaker_label(sid, speaker_map) for sid in seg_speaker_ids
                ],
                "start": seg["start"],
                "end": seg["end"],
                "start_hms": format_timestamp_hms(seg["start"]),
                "end_hms": format_timestamp_hms(seg["end"]),
                "text": to_traditional(seg["text"], converter),
            }
            f.write("    ")
            json.dump(item, f, ensure_ascii=False)
            if i < len(segments):
                f.write(",")
            f.write("\n")
        f.write("  ]\n")
        f.write("}\n")
    _atomic_write_text(output_path, _write)
    logger.info("✅ JSON 結構化輸出已儲存：%s", output_path)


# ──────────────────────────────────────────────
# 命令列參數
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="faster-whisper + pyannote.audio 語音轉文字 + 講者辨識"
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="音訊檔案或 glob pattern；支援遞迴 **（例如 recordings/**/*.m4a）",
    )
    parser.add_argument(
        "--token",
        "-t",
        default=None,
        help="HuggingFace Token（建議改用 HF_TOKEN 環境變數，避免出現在命令列歷史）",
    )
    parser.add_argument("--model", "-m", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--device", "-d", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", "-c", default="auto",
                        choices=["auto", "float16", "int8_float16", "int8", "float32"])
    parser.add_argument("--language", "-l", default="zh")
    parser.add_argument("--num-speakers", "-n", type=int, default=None)
    parser.add_argument("--diarize-device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--diarize-gap", default="auto", metavar="秒數|auto")
    parser.add_argument("--speaker-names", "-s", default=None, metavar="名稱1,名稱2,...")
    parser.add_argument("--output-format", "-f", default="all",
                        choices=["txt", "srt", "vtt", "json", "all"],
                        help="輸出格式：txt/srt/vtt/json/all（all 會輸出全部格式）")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument(
        "--output-suffix",
        default="auto",
        choices=["auto", "none", "path-hash"],
        help="輸出檔名後綴策略：auto=同批同名時自動加 hash；path-hash=一律加；none=不加",
    )
    parser.add_argument("--title", default="會議逐字稿",
                        help="TXT 輸出標題（預設：會議逐字稿）")
    parser.add_argument("--beam-size", type=int, default=5)

    wt_group = parser.add_mutually_exclusive_group()
    wt_group.add_argument("--word-timestamps", dest="word_timestamps",
                          action="store_true", default=True,
                          help="啟用字詞級時間戳（預設：開啟，講者對齊更精準）")
    wt_group.add_argument("--no-word-timestamps", dest="word_timestamps",
                          action="store_false",
                          help="關閉字詞時間戳，改用 segment 級對齊（較快但講者準確度略低）")

    parser.add_argument("--word-sentence-gap", type=float, default=0.8, metavar="秒數",
                        help="字詞對齊：同講者「輕停頓＋句末符號」切段門檻（預設 0.8 秒）")
    parser.add_argument("--word-hard-gap", type=float, default=2.5, metavar="秒數",
                        help="字詞對齊：同講者「強制切段」靜音門檻（預設 2.5 秒，不論句末）")
    parser.add_argument(
        "--overlap-labels",
        action="store_true",
        help="輸出重疊語音多講者標籤（例如：講者 1 / 講者 2）",
    )
    parser.add_argument(
        "--overlap-min-ratio",
        type=float,
        default=0.35,
        metavar="0~1",
        help="重疊標籤門檻（與片段重疊比例，預設 0.35）",
    )
    parser.add_argument("--min-interjection-chars", type=int, default=4, metavar="字數",
                        help="字數 ≤ 此值的超短片段會被吸收進前後文（預設 4）；設為 0 可停用")
    parser.add_argument("--interjection-max-gap", type=float, default=0.8, metavar="秒數",
                        help="超短插話吸收時，允許與前後文合併的最大停頓（預設 0.8 秒）")

    interj_group = parser.add_mutually_exclusive_group()
    interj_group.add_argument(
        "--same-speaker-interjections",
        dest="same_speaker_interjections",
        action="store_true",
        help="超短插話只吸收同講者片段（預設）",
    )
    interj_group.add_argument(
        "--allow-cross-speaker-interjections",
        dest="same_speaker_interjections",
        action="store_false",
        help="允許跨講者吸收超短插話（較平滑，但可能吃掉換人標記）",
    )
    parser.set_defaults(same_speaker_interjections=True)

    parser.add_argument("--no-traditional", action="store_true")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目標輸出檔已存在則略過；僅檢查是否存在，不驗證內容完整性",
    )
    parser.add_argument("--cache-dir", default=".cache", metavar="DIR")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument(
        "--parallel-model-load",
        action="store_true",
        default=False,
        help="並行載入 Whisper 與 pyannote（較快但在部分 Windows/CUDA 環境可能不穩）",
    )
    parser.add_argument("--max-segment-sec", type=float, default=60.0, metavar="秒數")
    parser.add_argument(
        "--max-segment-hard-sec",
        type=float,
        default=120.0,
        metavar="秒數",
        help="同講者片段硬上限；若 ≤ --max-segment-sec 會自動調整為 2 倍",
    )
    return parser.parse_args()


AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac",
    ".ogg", ".opus", ".aac", ".wma", ".webm",
    ".mkv", ".mov", ".avi", ".ts", ".mts",
}

FORMAT_EXTS = {
    "txt": [".txt"],
    "srt": [".srt"],
    "vtt": [".vtt"],
    "json": [".json"],
    "all": [".txt", ".srt", ".vtt", ".json"],
}


def _cache_stem(audio_path: Path) -> str:
    try:
        key_path = audio_path.resolve()
    except Exception:
        key_path = audio_path.absolute()
    key = os.path.normcase(str(key_path))
    suffix = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"{audio_path.stem}_{suffix}"


def _path_dedupe_key(p: Path) -> str:
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p.absolute()
    return os.path.normcase(str(resolved))


def _stem_dedupe_key(stem: str) -> str:
    return os.path.normcase(stem)


def _output_stem_hash(audio_path: Path, length: int = 8) -> str:
    try:
        key_path = audio_path.resolve()
    except Exception:
        key_path = audio_path.absolute()
    key = os.path.normcase(str(key_path))
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:length]


def resolve_output_stem(audio_path: Path, suffix_mode: str, has_stem_conflict: bool) -> str:
    stem = audio_path.stem
    if suffix_mode == "path-hash":
        return f"{stem}_{_output_stem_hash(audio_path)}"
    if suffix_mode == "auto" and has_stem_conflict:
        return f"{stem}_{_output_stem_hash(audio_path)}"
    return stem


def _whisper_cache_params(args, runtime_cfg: RuntimeConfig) -> dict:
    return {
        "model": args.model,
        "language": args.language,
        "beam_size": int(args.beam_size),
        "word_timestamps": bool(args.word_timestamps),
        "device": runtime_cfg.device,
        "compute_type": runtime_cfg.compute_type,
    }


def _diarize_cache_params(args, runtime_cfg: RuntimeConfig) -> dict:
    return {
        "pipeline": "pyannote/speaker-diarization-3.1",
        "num_speakers": args.num_speakers,
        "diarize_device": runtime_cfg.diarize_device,
    }


# ──────────────────────────────────────────────
# 斷點續跑
# ──────────────────────────────────────────────

class CheckpointManager:
    def __init__(
        self,
        audio_path: Path,
        cache_dir: Path,
        no_cache: bool = False,
        whisper_params: dict | None = None,
        diarize_params: dict | None = None,
    ):
        self.audio_path = audio_path
        self.cache_dir  = cache_dir
        self.no_cache   = no_cache
        self._whisper_params = whisper_params or {}
        self._diarize_params = diarize_params or {}
        cache_stem = _cache_stem(audio_path)
        self._whisper_path = cache_dir / f"{cache_stem}.whisper.json"
        self._diarize_path = cache_dir / f"{cache_stem}.diarize.json"

    def _source_meta(self, params: dict | None = None) -> dict:
        stat = self.audio_path.stat()
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "params": params or {},
        }

    @staticmethod
    def _tmp_cache_path(path: Path) -> Path:
        # 僅供 clear() 清理舊版固定命名暫存檔（*.json.tmp）使用。
        return path.with_name(f"{path.name}.tmp")

    def _is_valid(self, path: Path, expected_params: dict) -> bool:
        if not path.exists():
            return False
        try:
            data    = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != CACHE_SCHEMA_VERSION:
                logger.debug(
                    "快取版本不相符（%s）：cache=%s current=%s",
                    path.name, data.get("version"), CACHE_SCHEMA_VERSION,
                )
                return False
            meta    = data.get("_source_meta", {})
            current = self._source_meta(expected_params)
            mtime_cached = float(meta.get("mtime", 0.0) or 0.0)
            mtime_now    = float(current["mtime"])
            if abs(mtime_cached - mtime_now) >= MTIME_EPSILON_SEC:
                return False
            if meta.get("size") != current["size"]:
                return False
            if meta.get("params", {}) != expected_params:
                logger.debug(
                    "快取參數不相符（%s）：cache=%s current=%s",
                    path.name, meta.get("params", {}), expected_params,
                )
                return False
            return True
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as e:
            logger.debug("快取內容無效（%s）：%s", path.name, e)
            return False
        except OSError as e:
            logger.warning("⚠️  快取讀取失敗（%s）：%s；將視為無效快取", path.name, e)
            return False

    def _write(self, path: Path, payload, params: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f"{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp = Path(f.name)
                json.dump({
                    "version": CACHE_SCHEMA_VERSION,
                    "_source_meta": self._source_meta(params),
                    "data": payload,
                },
                          f,
                          ensure_ascii=False,
                          indent=2)
            tmp.replace(path)
        except Exception:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            raise

    def _read(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))["data"]

    def load_whisper(self):
        if self.no_cache or not self._is_valid(self._whisper_path, self._whisper_params):
            return None
        logger.info("⚡ 復用 Whisper 快取：%s", self._whisper_path.name)
        return self._read(self._whisper_path)

    def save_whisper(self, segments: list) -> None:
        if self.no_cache:
            return
        # words 以 dict 儲存（非 Word 物件），_get_word_items 以 isinstance 分支相容兩種格式。
        # 這裡保留 [] 與 None 的語義差異：[] 代表啟用後無詞；None 代表沒有詞級資料。
        payload = [
            {
                "start": seg.start, "end": seg.end, "text": seg.text,
                "words": (
                    [{"start": w.start, "end": w.end, "word": w.word}
                     for w in seg.words]
                    if seg.words is not None else None
                ),
            }
            for seg in segments
        ]
        self._write(self._whisper_path, payload, self._whisper_params)
        logger.info("💾 Whisper 結果已快取：%s", self._whisper_path.name)

    def load_diarize(self):
        if self.no_cache or not self._is_valid(self._diarize_path, self._diarize_params):
            return None
        logger.info("⚡ 復用 pyannote 快取：%s", self._diarize_path.name)
        return self._read(self._diarize_path)

    def current_diarize_device(self, default: str = "cpu") -> str:
        device = self._diarize_params.get("diarize_device", default)
        return str(device) if device else default

    def update_diarize_device(self, device: str) -> None:
        """更新 diarize 裝置快取鍵，讓快取與實際執行裝置一致。"""
        prev = self._diarize_params.get("diarize_device")
        if prev != device:
            logger.debug("   diarize 快取裝置更新：%s -> %s", prev, device)
            self._diarize_params = {**self._diarize_params, "diarize_device": device}

    def save_diarize(self, segments: list) -> None:
        if self.no_cache:
            return
        self._write(self._diarize_path, segments, self._diarize_params)
        logger.info("💾 pyannote 結果已快取：%s", self._diarize_path.name)

    def clear(self) -> None:
        for p in (self._whisper_path, self._diarize_path):
            # 清理目前命名（*.json.tmp）；p.with_suffix(".tmp") 為 v7 前舊命名，
            # 可於確認所有部署均已升級後移除。
            for tmp in {self._tmp_cache_path(p), p.with_suffix(".tmp")}:
                tmp.unlink(missing_ok=True)
            if p.exists():
                p.unlink()
                logger.info("🗑️  已清除快取：%s", p.name)


class _SegmentAdapter:
    __slots__ = ("start", "end", "text", "words")
    def __init__(self, d: dict):
        self.start = d["start"]
        self.end   = d["end"]
        self.text  = d["text"]
        self.words = d.get("words")


def restore_whisper_segments(cached: list) -> list:
    return [_SegmentAdapter(d) for d in cached]


def resolve_audio_files(patterns: list) -> list:
    seen = set()
    result = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern, recursive=True))
            if not matches:
                logger.warning("⚠️  glob pattern 沒有找到任何符合的檔案：%s", pattern)
            for m in matches:
                p = Path(m)
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                    key = _path_dedupe_key(p)
                    if key not in seen:
                        seen.add(key); result.append(p)
        else:
            p = Path(pattern)
            if not p.exists():
                logger.warning("⚠️  找不到檔案：%s", p); continue
            if not p.is_file():
                logger.warning("⚠️  路徑不是檔案（略過）：%s", p); continue
            if p.suffix.lower() not in AUDIO_EXTENSIONS:
                logger.warning("⚠️  不支援的副檔名（略過）：%s", p); continue
            key = _path_dedupe_key(p)
            if key not in seen:
                seen.add(key); result.append(p)
    return result


# ──────────────────────────────────────────────
# 單檔處理
# ──────────────────────────────────────────────

def _print_error_hint(err: str):
    e = err.lower()
    if "token" in e or "401" in e:
        logger.error("   → Token 錯誤，請確認 --token 參數是否正確")
    elif "gated" in e or "403" in e:
        logger.error("   → 模型存取被拒，請至 HuggingFace 接受使用條款：")
        logger.error("     https://huggingface.co/pyannote/speaker-diarization-3.1")
    elif "ffmpeg" in e:
        logger.error("   → FFmpeg 找不到，請確認已加入系統 PATH")
    elif _is_cudnn_symbol_error(e):
        logger.error("   → cuDNN DLL 版本衝突（cudnnGetLibConfig）")
        logger.error("     建議先移除 --diarize-device cuda，改用 --diarize-device auto 或 cpu")
    elif any(k in e for k in ("out of memory", "oom", "cuda out", "memory")):
        logger.error("   → 記憶體不足，請將 --model 改為較小的模型（如 small）")


def _cleanup_pending_wav(
    audio_path: Path,
    pending_wav: str | None,
    wav_future: Future | None,
    pending_wav_is_new: bool = True,
) -> None:
    """清理 WAV 暫存檔。

    pending_wav_is_new=True  → 本次新建，執行刪除。
    pending_wav_is_new=False → 復用既有或來源本身，保留不刪（供下次復用）。
    預設值為 True（保守），確保在例外路徑中不遺留垃圾檔案。
    """
    # Whisper 若先失敗，仍嘗試從 wav_future 取回轉檔路徑，避免遺留暫存檔
    if pending_wav is None and wav_future is not None:
        try:
            if wav_future.done() and not wav_future.cancelled():
                result = wav_future.result()
                if isinstance(result, tuple):
                    # convert_to_wav 回傳 (path, newly_created)
                    maybe_wav, pending_wav_is_new = result
                elif isinstance(result, str):
                    # 相容舊格式（不應出現，但保守處理）
                    maybe_wav = result
                else:
                    maybe_wav = None
                if maybe_wav:
                    pending_wav = maybe_wav
        except Exception:
            pass

    # 只有「本次新建」的暫存檔才刪除；復用或來源本身保留供下次使用
    if pending_wav and pending_wav_is_new:
        pending = Path(pending_wav)
        if pending.exists():
            try:
                same_as_source = pending.resolve() == audio_path.resolve()
            except Exception:
                same_as_source = pending == audio_path
            if not same_as_source:
                pending.unlink(missing_ok=True)


def _run_stages(
    audio_path: Path,
    whisper_model: WhisperModel,
    pipeline_state: PipelineState,
    args,
    ckpt: CheckpointManager,
) -> tuple[list, list, str]:
    _pending_wav: str | None = None
    _pending_wav_is_new: bool = False     # 與 convert_to_wav 回傳的 newly_created 對應
    wav_path: Path | None = None
    wav_future: Future | None = None

    try:
        cached_whisper = ckpt.load_whisper()
        cached_diarize = ckpt.load_diarize()
        need_whisper   = cached_whisper is None
        need_wav       = cached_diarize is None
        actual_diarize_device = ckpt.current_diarize_device(default=args.diarize_device)

        if need_whisper and need_wav:
            logger.info("\n⚡ 並行執行：Whisper 辨識 ＋ WAV 轉換...")
            with ThreadPoolExecutor(max_workers=2) as ex:
                wav_future       = ex.submit(convert_to_wav, str(audio_path))
                whisper_segments = run_transcription(
                    whisper_model, str(audio_path),
                    language=args.language, beam_size=args.beam_size,
                    word_timestamps=args.word_timestamps,
                )
                _pending_wav, _pending_wav_is_new = wav_future.result()  # 解包 tuple
            wav_path = Path(_pending_wav)
            ckpt.save_whisper(whisper_segments)

        elif need_whisper:
            # need_wav=False 代表 diarize 已快取，後續不需要 wav_path。
            logger.info("\n🎙️  執行 Whisper 辨識...")
            whisper_segments = run_transcription(
                whisper_model, str(audio_path),
                language=args.language, beam_size=args.beam_size,
                word_timestamps=args.word_timestamps,
            )
            ckpt.save_whisper(whisper_segments)

        elif need_wav:
            logger.info("\n📂 Whisper 已快取，僅轉換 WAV...")
            whisper_segments = restore_whisper_segments(cached_whisper)
            _pending_wav, _pending_wav_is_new = convert_to_wav(str(audio_path))  # 解包 tuple
            wav_path = Path(_pending_wav)

        else:
            logger.info("\n⚡ 兩者均已快取，直接載入...")
            whisper_segments = restore_whisper_segments(cached_whisper)

        if cached_diarize is not None:
            speaker_segments = cached_diarize
        else:
            if wav_path is None:
                raise RuntimeError(
                    f"internal error: WAV path missing before diarization ({audio_path.name})"
                )
            try:
                speaker_segments = run_diarization_with_pipeline(
                    str(wav_path), pipeline_state.pipeline, args.num_speakers
                )
            except Exception as diarize_err:
                if _is_cudnn_symbol_error(diarize_err):
                    logger.warning(
                        "⚠️  diarization 發生 cuDNN 符號錯誤，改用 CPU 重試一次..."
                    )
                    pipeline_state.pipeline = pipeline_state.pipeline.to(torch.device("cpu"))
                    actual_diarize_device = "cpu"
                    speaker_segments = run_diarization_with_pipeline(
                        str(wav_path), pipeline_state.pipeline, args.num_speakers
                    )
                else:
                    raise
        if cached_diarize is None:
            ckpt.update_diarize_device(actual_diarize_device)
            ckpt.save_diarize(speaker_segments)

        return whisper_segments, speaker_segments, actual_diarize_device
    finally:
        # 把 WAV 暫存生命週期收斂在 stage 內，避免外層拿不到中間狀態時漏清理。
        try:
            _cleanup_pending_wav(
                audio_path, _pending_wav, wav_future, _pending_wav_is_new
            )
        except Exception:
            pass


def process_single_file(
    audio_path: Path,
    whisper_model: WhisperModel,
    pipeline_state: PipelineState,
    output_dir: Path,
    output_stem: str,
    batch_cfg: BatchConfig,
) -> tuple[bool, RuntimeConfig]:
    args = batch_cfg.args
    converter = batch_cfg.converter
    cache_dir = batch_cfg.cache_dir
    runtime_cfg = batch_cfg.runtime_cfg

    ckpt = CheckpointManager(
        audio_path,
        cache_dir,
        no_cache=args.no_cache,
        whisper_params=_whisper_cache_params(args, runtime_cfg),
        diarize_params=_diarize_cache_params(args, runtime_cfg),
    )

    if args.skip_existing:
        fmt = args.output_format
        exts = FORMAT_EXTS[fmt]
        if all((output_dir / f"{output_stem}{e}").exists() for e in exts):
            logger.info("⏭️  略過（輸出已存在）：%s", audio_path.name)
            return True, runtime_cfg

    try:
        whisper_segments, speaker_segments, actual_diarize_device = _run_stages(
            audio_path,
            whisper_model,
            pipeline_state,
            args,
            ckpt,
        )
        if actual_diarize_device != runtime_cfg.diarize_device:
            runtime_cfg = replace(runtime_cfg, diarize_device=actual_diarize_device)

        if args.verbose:
            logger.debug("\n📋 Whisper 預覽（前 5）：")
            for seg in whisper_segments[:5]:
                logger.debug("  [%s] %s", format_timestamp_hms(seg.start), seg.text.strip())

        diarize_gap = resolve_diarize_gap(args.diarize_gap, speaker_segments)

        use_word_align = args.word_timestamps
        if use_word_align:
            has_words = any(
                getattr(s, "words", None) is not None for s in whisper_segments
            )
            if not has_words:
                logger.warning(
                    "⚠️  Whisper 快取不含字詞時間戳，本次降回 segment 級對齊。\n"
                    "   解法：加上 --clear-cache 重新轉錄，即可啟用字詞級對齊。"
                )
                use_word_align = False

        if use_word_align:
            logger.info("\n🔗 整合（字詞級對齊）...")
            combined = assign_speakers_word_level(
                whisper_segments, speaker_segments,
                gap_sec=diarize_gap,
                sentence_split_gap=args.word_sentence_gap,
                hard_split_gap=args.word_hard_gap,
                overlap_labels=args.overlap_labels,
                overlap_ratio_min=args.overlap_min_ratio,
            )
        else:
            logger.info("\n🔗 整合（segment 級對齊）...")
            combined = assign_speakers(
                whisper_segments, speaker_segments, gap_sec=diarize_gap,
                overlap_labels=args.overlap_labels, overlap_ratio_min=args.overlap_min_ratio,
            )

        combined = merge_consecutive(
            combined,
            max_duration_sec=args.max_segment_sec,
            max_duration_hard_sec=args.max_segment_hard_sec,
        )
        if args.min_interjection_chars > 0:
            combined = absorb_short_interjections(
                combined,
                min_chars=args.min_interjection_chars,
                same_speaker_only=args.same_speaker_interjections,
                max_gap_sec=args.interjection_max_gap,
            )
        logger.info("✅ 整合完成！共 %d 個對話片段", len(combined))

        speaker_map = build_speaker_map(combined, args.speaker_names)

        logger.info("")
        fmt = args.output_format
        if fmt in ("txt", "all"):
            save_txt(combined, output_dir / f"{output_stem}.txt", converter,
                     str(audio_path), speaker_map, show_timestamps=not args.no_timestamps,
                     title=args.title)
        if fmt in ("srt", "all"):
            save_srt(combined, output_dir / f"{output_stem}.srt", converter, speaker_map)
        if fmt in ("vtt", "all"):
            save_vtt(combined, output_dir / f"{output_stem}.vtt", converter, speaker_map)
        if fmt in ("json", "all"):
            save_json(combined, output_dir / f"{output_stem}.json", converter,
                      str(audio_path), speaker_map)

        logger.debug("\n📋 預覽（前 8）：")
        logger.debug("-" * 60)
        for seg in combined[:8]:
            label = resolve_segment_speaker_label(seg, speaker_map)
            text  = to_traditional(seg["text"], converter)
            logger.debug("【%s】[%s] %s", label, format_timestamp_hms(seg["start"]), text)
        if len(combined) > 8:
            logger.debug("  ... 還有 %d 個，請開啟輸出檔案查看", len(combined) - 8)

        return True, runtime_cfg

    except Exception as e:
        logger.error("\n❌ 處理失敗：%s\n   錯誤：%s", audio_path.name, e)
        _print_error_hint(str(e))
        return False, runtime_cfg


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.verbose)
    if args.overlap_min_ratio <= 0 or args.overlap_min_ratio > 1:
        logger.error("❌ --overlap-min-ratio 必須在 (0, 1] 範圍內")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("❌ 請提供 HuggingFace Token（--token 或 HF_TOKEN 環境變數）")
        sys.exit(1)
    # 最佳努力縮短 token 可見時間；注意 Python 無法保證立即抹除字串記憶體內容。
    args.token = None

    print("=" * 60)
    print("  faster-whisper + pyannote 語音轉文字 + 講者辨識")
    print("=" * 60)

    if args.word_timestamps:
        logger.info("ℹ️  字詞時間戳：開啟（預設）→ 字詞級講者對齊")
    else:
        logger.info("ℹ️  字詞時間戳：已關閉（--no-word-timestamps）→ segment 級對齊")
    if args.overlap_labels:
        logger.info("ℹ️  重疊語音標籤：開啟（門檻 %.2f）", args.overlap_min_ratio)
    else:
        logger.info("ℹ️  重疊語音標籤：關閉（可用 --overlap-labels 開啟）")

    try:
        check_ffmpeg()
    except RuntimeError as e:
        logger.error("❌ %s", e); sys.exit(1)

    audio_files = resolve_audio_files(args.audio)
    if not audio_files:
        logger.error("❌ 沒有找到有效音訊檔案"); sys.exit(1)

    total = len(audio_files)
    logger.info("\n📂 共找到 %d 個音訊檔案：", total)
    for f in audio_files:
        logger.info("   • %s", f)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("\nℹ️  自動偵測裝置：%s", device.upper())

    compute_type = detect_compute_type(device, args.compute_type)
    if not HAS_OPENCC and not args.no_traditional:
        logger.warning("⚠️  未安裝 opencc-python-reimplemented，將略過簡轉繁步驟")
        logger.warning("   建議安裝：pip install opencc-python-reimplemented")
    converter    = opencc.OpenCC("s2twp") if HAS_OPENCC and not args.no_traditional else None
    t_start      = time.time()
    cache_dir    = Path(args.cache_dir)

    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("📁 快取目錄：%s", cache_dir.resolve())

    resolved_diarize_device = args.diarize_device
    try:
        if args.parallel_model_load:
            logger.info("\n⚡ 並行載入兩個模型（Whisper ＋ pyannote）...")
            logger.info("─" * 60)
            with ThreadPoolExecutor(max_workers=2) as executor:
                pipeline_future: Future = executor.submit(
                    load_diarization_pipeline, token, device, args.diarize_device)
                whisper_model = load_whisper_model(args.model, device, compute_type)
                logger.info("\n⏳ 等待 pyannote 載入完成...")
                pipeline, resolved_diarize_device = pipeline_future.result()
        else:
            logger.info("\n🔄 依序載入兩個模型（避免 Windows CUDA DLL 競態）...")
            logger.info("─" * 60)
            pipeline, resolved_diarize_device = load_diarization_pipeline(
                token, device, args.diarize_device
            )
            whisper_model = load_whisper_model(args.model, device, compute_type)
    except Exception as e:
        logger.error("\n❌ 模型初始化失敗：%s", e)
        _print_error_hint(str(e)); sys.exit(1)
    finally:
        # 最佳努力解除參照；原字串仍可能存活至 GC 回收（Python 限制）。
        token = None

    runtime_cfg = RuntimeConfig(
        device=device,
        compute_type=compute_type,
        diarize_device=resolved_diarize_device,
    )
    batch_cfg = BatchConfig(
        args=args,
        converter=converter,
        cache_dir=cache_dir,
        runtime_cfg=runtime_cfg,
    )
    logger.info("\n✅ 兩個模型均已就緒，開始批次處理\n")
    pipeline_state = PipelineState(pipeline=pipeline)
    results = []
    recent_elapsed: deque[float] = deque(maxlen=5)

    # ── 修正：衝突偵測改以「實際輸出目錄 + stem」為鍵，
    #         不論是否指定 --output-dir 均會正確偵測同目錄同名衝突。
    # 舊邏輯只在 args.output_dir 存在時才執行，導致預設輸出到來源目錄時，
    # 同目錄下 meeting.mp3 / meeting.wav 會互相覆蓋輸出檔案。
    _planned_counts: dict[tuple[str, str], int] = {}
    for audio_path in audio_files:
        _out_dir = (Path(args.output_dir).resolve()
                    if args.output_dir
                    else audio_path.parent.resolve())
        _key = (os.path.normcase(str(_out_dir)), _stem_dedupe_key(audio_path.stem))
        _planned_counts[_key] = _planned_counts.get(_key, 0) + 1
    conflict_pairs: set[tuple[str, str]] = {
        k for k, v in _planned_counts.items() if v > 1
    }
    if conflict_pairs and args.output_suffix == "auto":
        logger.info(
            "ℹ️  偵測到同目錄同名檔案（%d 組），--output-suffix auto 將自動加上 path-hash",
            len(conflict_pairs),
        )
    elif conflict_pairs and args.output_suffix == "none":
        logger.warning(
            "⚠️  偵測到同目錄同名檔案（%d 組）且 --output-suffix=none，可能覆蓋既有輸出",
            len(conflict_pairs),
        )

    for idx, audio_path in enumerate(audio_files, start=1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{total}] 處理中：{audio_path.name}")
        print(f"{'═' * 60}")

        output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # ── 修正：查詢鍵改用 (resolved_output_dir, stem)，與上方 conflict_pairs 一致。
        _out_dir_key = os.path.normcase(str(output_dir.resolve()))
        output_stem = resolve_output_stem(
            audio_path,
            suffix_mode=args.output_suffix,
            has_stem_conflict=(_out_dir_key, _stem_dedupe_key(audio_path.stem)) in conflict_pairs,
        )
        if output_stem != audio_path.stem:
            logger.info("ℹ️  輸出檔名：%s.* -> %s.*", audio_path.stem, output_stem)

        if args.clear_cache:
            CheckpointManager(audio_path, cache_dir, no_cache=False).clear()

        t_file = time.time()
        ok, updated_runtime_cfg = process_single_file(
            audio_path=audio_path,
            whisper_model=whisper_model,
            pipeline_state=pipeline_state,
            output_dir=output_dir,
            output_stem=output_stem,
            batch_cfg=batch_cfg,
        )
        if updated_runtime_cfg != batch_cfg.runtime_cfg:
            logger.info(
                "ℹ️  偵測到 pyannote 裝置切換，後續快取鍵改用：%s",
                updated_runtime_cfg.diarize_device.upper(),
            )
            batch_cfg.runtime_cfg = updated_runtime_cfg
        elapsed = time.time() - t_file
        recent_elapsed.append(elapsed)
        logger.info("\n%s  %s（耗時 %.1f 秒）", "✅ 成功" if ok else "❌ 失敗",
                    audio_path.name, elapsed)
        if idx < total:
            avg_sec = sum(recent_elapsed) / len(recent_elapsed)
            eta_sec = avg_sec * (total - idx)
            logger.info("   預估剩餘：%.0f 秒", eta_sec)
        results.append((audio_path, ok))

    elapsed_total = time.time() - t_start
    success = sum(1 for _, ok in results if ok)
    failed  = total - success

    print(f"\n{'═' * 60}")
    print(f"  批次處理完成｜總耗時 {elapsed_total:.1f} 秒")
    print(f"  成功：{success} 個　失敗：{failed} 個　共計：{total} 個")
    print(f"{'═' * 60}")

    if failed:
        print("\n❌ 以下檔案處理失敗：")
        for path, ok in results:
            if not ok:
                print(f"   • {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()