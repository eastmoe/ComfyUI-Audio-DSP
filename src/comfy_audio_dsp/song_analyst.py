from __future__ import annotations

import json
import re
from typing import Any

import torch

from .common import audio_waveform, copy_audio
from .pitch_time import PITCH_KEYS, PITCH_SCALES

FLAT_TO_SHARP = {
    "DB": "C#",
    "EB": "D#",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}


def _loads_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        data = json.loads(str(text))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1)[0].item()
        return float(value)
    except Exception:
        return float(fallback)


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        return _float_list(parsed)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    return []


def _int_list(value: Any) -> list[int]:
    return [int(round(v)) for v in _float_list(value)]


def _segments(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = _loads_json(value)
        if parsed:
            return _segments(parsed.get("segments") or parsed.get("similar_segments"))
        try:
            return _segments(json.loads(value))
        except Exception:
            return []
    if isinstance(value, dict):
        return _segments(value.get("segments") or value.get("similar_segments"))
    if isinstance(value, (list, tuple)):
        out = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            start = _to_float(item.get("start", item.get("start_s", item.get("representative_start"))), 0.0)
            end = _to_float(item.get("end", item.get("end_s", item.get("representative_end"))), 0.0)
            label = str(item.get("label", item.get("name", f"segment_{index}")))
            if end > start:
                out.append({"index": index, "start": start, "end": end, "label": label})
        return out
    return []


def song_bpm_value(default_bpm: float, song_bpm: Any = None, analysis_json: str | None = None) -> float:
    bpm = _to_float(song_bpm, 0.0)
    if bpm > 0.0:
        return bpm
    data = _loads_json(analysis_json)
    bpm = _to_float(data.get("bpm"), 0.0)
    return bpm if bpm > 0.0 else float(default_bpm)


def song_key_to_pitch_controls(key_text: str = "", analysis_json: str | None = None) -> tuple[str, str, str]:
    data = _loads_json(analysis_json)
    label = str(key_text or data.get("key") or "").strip()
    match = re.search(r"([A-Ga-g])\s*([#bB]?)(?:\s*[-_:]?\s*(major|minor|maj|min|chromatic))?", label)
    if not match:
        result = {"input": label, "key": "C", "scale": "chromatic", "status": "fallback"}
        return "C", "chromatic", json.dumps(result, ensure_ascii=False)

    letter = match.group(1).upper()
    accidental = match.group(2)
    if accidental == "#":
        root = f"{letter}#"
    elif accidental.lower() == "b":
        root = FLAT_TO_SHARP.get(f"{letter}B", letter)
    else:
        root = letter
    if root not in PITCH_KEYS:
        root = "C"
    mode = (match.group(3) or "major").lower()
    if mode == "maj":
        mode = "major"
    if mode == "min":
        mode = "minor"
    scale = mode if mode in PITCH_SCALES else "chromatic"
    result = {"input": label, "key": root, "scale": scale, "status": "ok"}
    return root, scale, json.dumps(result, ensure_ascii=False)


def analysis_to_dsp_controls(analysis_json: str) -> tuple[float, str, str, str, list[float], list[int], list[dict[str, Any]], str]:
    data = _loads_json(analysis_json)
    key_text = str(data.get("key") or "")
    key, scale, key_details = song_key_to_pitch_controls(key_text, analysis_json)
    beat_times = _float_list(data.get("beat_times") or data.get("chord_times") or data.get("boundary_times"))
    downbeats = _int_list(data.get("downbeats"))
    segments = _segments(data.get("segments") or data.get("similar_segments"))
    bpm = song_bpm_value(120.0, data.get("bpm"), analysis_json)
    details = {
        "bpm": bpm,
        "key_text": key_text,
        "key_details": json.loads(key_details),
        "beat_count": len(beat_times),
        "downbeat_count": sum(1 for value in downbeats if value),
        "segment_count": len(segments),
    }
    return bpm, key_text, key, scale, beat_times, downbeats, segments, json.dumps(details, ensure_ascii=False)


def song_segment_selector(
    audio: dict,
    segment_index: int,
    label_filter: str,
    padding_s: float,
    song_segments: Any = None,
    analysis_json: str | None = None,
) -> tuple[dict, float, float, str]:
    waveform, sample_rate = audio_waveform(audio)
    segments = _segments(song_segments) or _segments(_loads_json(analysis_json))
    if label_filter.strip():
        needle = label_filter.strip().lower()
        segments = [item for item in segments if needle in str(item.get("label", "")).lower()]
    if not segments:
        details = {"status": "no_segments", "start_s": 0.0, "end_s": waveform.shape[-1] / max(float(sample_rate), 1.0)}
        return copy_audio(audio, waveform), 0.0, float(details["end_s"]), json.dumps(details, ensure_ascii=False)

    index = max(0, min(int(segment_index), len(segments) - 1))
    item = segments[index]
    pad = max(0.0, float(padding_s))
    duration = waveform.shape[-1] / max(float(sample_rate), 1.0)
    start_s = max(0.0, _to_float(item.get("start"), 0.0) - pad)
    end_s = min(duration, _to_float(item.get("end"), duration) + pad)
    start = max(0, min(waveform.shape[-1] - 1, int(round(start_s * sample_rate))))
    end = max(start + 1, min(waveform.shape[-1], int(round(end_s * sample_rate))))
    details = {"status": "ok", "selected": item, "start_s": start_s, "end_s": end_s, "available_segments": len(segments)}
    return copy_audio(audio, waveform[..., start:end]), start_s, end_s, json.dumps(details, ensure_ascii=False)


def song_beat_grid_slicer(
    audio: dict,
    beats_per_slice: int,
    padding_ms: float,
    prefer_downbeats: bool,
    beat_times: Any = None,
    downbeats: Any = None,
    analysis_json: str | None = None,
) -> tuple[dict, str, int]:
    waveform, sample_rate = audio_waveform(audio)
    data = _loads_json(analysis_json)
    beats = _float_list(beat_times) or _float_list(data.get("beat_times"))
    downbeat_flags = _int_list(downbeats) or _int_list(data.get("downbeats"))
    duration = waveform.shape[-1] / max(float(sample_rate), 1.0)
    if len(beats) < 2:
        return copy_audio(audio, waveform), "[]", 0

    starts = [idx for idx in range(len(beats) - 1) if not prefer_downbeats or (idx < len(downbeat_flags) and downbeat_flags[idx])]
    if not starts:
        starts = list(range(0, len(beats) - 1, max(1, int(beats_per_slice))))
    pad = max(0.0, float(padding_ms) / 1000.0)
    step = max(1, int(beats_per_slice))
    slices = []
    for slice_index, beat_index in enumerate(starts):
        end_index = min(len(beats) - 1, beat_index + step)
        start_s = max(0.0, float(beats[beat_index]) - pad)
        end_s = min(duration, float(beats[end_index]) + pad)
        if end_s <= start_s:
            continue
        slices.append({"index": slice_index, "beat_index": beat_index, "start_s": round(start_s, 4), "end_s": round(end_s, 4)})
    return copy_audio(audio, waveform), json.dumps(slices, ensure_ascii=False), len(slices)
