from __future__ import annotations

import math

import numpy as np
import torch
from scipy import signal

from .common import AUDIO_EPS, audio_waveform, clamp01, copy_audio, db_to_amp, mix_audio

PITCH_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_SCALES = ["chromatic", "major", "minor", "major_pentatonic", "minor_pentatonic"]
SCALE_INTERVALS = {
    "chromatic": tuple(range(12)),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
}


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _from_numpy(data: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(data.astype(np.float32, copy=False)).to(device=like.device, dtype=like.dtype)


def _fit_length(data: np.ndarray, length: int) -> np.ndarray:
    if data.shape[-1] == length:
        return data.astype(np.float32, copy=False)
    if data.shape[-1] <= 1 or length <= 1:
        return np.zeros(length, dtype=np.float32)
    return signal.resample(data, length).astype(np.float32)


def _phase_vocoder_row(x: np.ndarray, stretch: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    stretch = max(0.05, float(stretch))
    if x.shape[-1] < 32 or abs(stretch - 1.0) < 1.0e-4:
        return x.astype(np.float32, copy=True)
    n_fft = min(n_fft, max(32, 2 ** int(math.floor(math.log2(max(32, x.shape[-1]))))))
    hop = max(1, min(hop, n_fft // 4))
    _, _, stft = signal.stft(x, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    if stft.shape[-1] < 2:
        return _fit_length(x, max(1, int(round(x.shape[-1] * stretch))))

    rate = 1.0 / stretch
    time_steps = np.arange(0.0, stft.shape[-1] - 1, rate, dtype=np.float64)
    omega = 2.0 * np.pi * hop * np.arange(stft.shape[0], dtype=np.float64) / float(n_fft)
    phase = np.angle(stft[:, 0]).astype(np.float64)
    out = np.empty((stft.shape[0], len(time_steps)), dtype=np.complex64)
    for index, step in enumerate(time_steps):
        left = int(math.floor(float(step)))
        frac = float(step) - left
        mag = (1.0 - frac) * np.abs(stft[:, left]) + frac * np.abs(stft[:, left + 1])
        out[:, index] = mag * np.exp(1j * phase)
        delta = np.angle(stft[:, left + 1]) - np.angle(stft[:, left]) - omega
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase += omega + delta

    _, y = signal.istft(out, nperseg=n_fft, noverlap=n_fft - hop, input_onesided=True)
    return _fit_length(y.astype(np.float32), max(1, int(round(x.shape[-1] * stretch))))


def _apply_rows(data: np.ndarray, func) -> np.ndarray:
    rows = data.reshape(-1, data.shape[-1])
    out_rows = [func(row) for row in rows]
    length = max(row.shape[-1] for row in out_rows)
    out = np.zeros((len(out_rows), length), dtype=np.float32)
    for index, row in enumerate(out_rows):
        out[index, : row.shape[-1]] = row
    return out.reshape(*data.shape[:-1], length)


def _classic_resample_row(x: np.ndarray, speed_ratio: float) -> np.ndarray:
    speed_ratio = max(0.05, float(speed_ratio))
    target_length = max(1, int(round(x.shape[-1] / speed_ratio)))
    return _fit_length(x, target_length)


def _pitch_shift_row(x: np.ndarray, semitones: float) -> np.ndarray:
    ratio = 2.0 ** (float(semitones) / 12.0)
    sped = _classic_resample_row(x, ratio)
    shifted = _phase_vocoder_row(sped, ratio)
    return _fit_length(shifted, x.shape[-1])


def pitch_shifter(audio: dict, semitones: float, cents: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    total_semitones = float(semitones) + float(cents) / 100.0
    wet_np = _apply_rows(_to_numpy(waveform), lambda row: _pitch_shift_row(row, total_semitones))
    wet = _from_numpy(wet_np, waveform)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def time_stretcher(audio: dict, time_ratio: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    wet_np = _apply_rows(_to_numpy(waveform), lambda row: _phase_vocoder_row(row, time_ratio))
    wet = _from_numpy(wet_np, waveform)
    dry = _from_numpy(_apply_rows(_to_numpy(waveform), lambda row: _fit_length(row, wet_np.shape[-1])), waveform)
    return copy_audio(audio, mix_audio(dry, wet, mix))


def resampler_classic(audio: dict, speed_ratio: float, output_gain_db: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    wet_np = _apply_rows(_to_numpy(waveform), lambda row: _classic_resample_row(row, speed_ratio))
    return copy_audio(audio, _from_numpy(wet_np, waveform) * float(db_to_amp(output_gain_db)))


def harmonizer(
    audio: dict,
    voice_1_semitones: float,
    voice_1_gain: float,
    voice_2_semitones: float,
    voice_2_gain: float,
    voice_3_semitones: float,
    voice_3_gain: float,
    voice_4_semitones: float,
    voice_4_gain: float,
    dry_gain: float,
    mix: float,
) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    source = _to_numpy(waveform)
    voices = [
        (voice_1_semitones, voice_1_gain),
        (voice_2_semitones, voice_2_gain),
        (voice_3_semitones, voice_3_gain),
        (voice_4_semitones, voice_4_gain),
    ]
    wet_np = source * float(dry_gain)
    for semitones, gain in voices:
        if abs(float(gain)) > 1.0e-5:
            wet_np += _apply_rows(source, lambda row, st=semitones: _pitch_shift_row(row, st)) * float(gain)
    wet_np /= max(1.0, abs(float(dry_gain)) + sum(abs(float(gain)) for _st, gain in voices))
    wet = _from_numpy(wet_np, waveform)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def _estimate_pitch(frame: np.ndarray, sample_rate: int, min_hz: float = 50.0, max_hz: float = 1600.0) -> float:
    frame = frame.astype(np.float64, copy=False)
    frame = frame - np.mean(frame)
    energy = float(np.sqrt(np.mean(frame * frame) + AUDIO_EPS))
    if energy < 1.0e-4:
        return 0.0
    corr = signal.correlate(frame, frame, mode="full", method="fft")[len(frame) - 1 :]
    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(len(corr) - 1, int(sample_rate / min_hz))
    if max_lag <= min_lag:
        return 0.0
    segment = corr[min_lag:max_lag]
    lag = int(np.argmax(segment)) + min_lag
    clarity = corr[lag] / max(corr[0], AUDIO_EPS)
    if clarity < 0.2:
        return 0.0
    return float(sample_rate) / float(lag)


def _nearest_scale_frequency(frequency: float, key: str, scale: str) -> float:
    if frequency <= 0.0:
        return frequency
    key_offset = PITCH_KEYS.index(key) if key in PITCH_KEYS else 0
    intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS["chromatic"])
    midi = 69.0 + 12.0 * math.log2(frequency / 440.0)
    candidates = []
    base_octave = int(math.floor((midi - key_offset) / 12.0))
    for octave in range(base_octave - 1, base_octave + 2):
        for interval in intervals:
            candidates.append(octave * 12 + key_offset + interval)
    target = min(candidates, key=lambda note: abs(note - midi))
    return 440.0 * (2.0 ** ((target - 69.0) / 12.0))


def _pitch_correct_row(x: np.ndarray, sample_rate: int, key: str, scale: str, correction_speed: float, mix: float) -> np.ndarray:
    amount = clamp01(correction_speed)
    if amount <= 0.0:
        return x.astype(np.float32, copy=True)
    frame_size = min(4096, max(512, 2 ** int(math.floor(math.log2(max(512, min(len(x), 4096)))))))
    hop = frame_size // 4
    window = np.hanning(frame_size).astype(np.float32)
    padded = np.pad(x.astype(np.float32), (frame_size, frame_size), mode="constant")
    out = np.zeros_like(padded, dtype=np.float32)
    weight = np.zeros_like(padded, dtype=np.float32)
    previous_ratio = 1.0
    smoothing = math.exp(-amount * 4.0)
    for start in range(0, len(padded) - frame_size + 1, hop):
        frame = padded[start : start + frame_size] * window
        pitch = _estimate_pitch(frame, sample_rate)
        if pitch > 0.0:
            target = _nearest_scale_frequency(pitch, key, scale)
            desired_ratio = target / max(pitch, AUDIO_EPS)
            ratio = smoothing * previous_ratio + (1.0 - smoothing) * desired_ratio
            previous_ratio = ratio
            shifted = _classic_resample_row(frame, ratio)
            shifted = _fit_length(shifted, frame_size)
            processed = frame * (1.0 - float(mix)) + shifted * float(mix)
        else:
            processed = frame
        out[start : start + frame_size] += processed * window
        weight[start : start + frame_size] += window * window
    out = out / np.maximum(weight, 1.0e-5)
    return out[frame_size : frame_size + len(x)].astype(np.float32)


def pitch_correction(audio: dict, key: str, scale: str, correction_speed: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet_np = _apply_rows(_to_numpy(waveform), lambda row: _pitch_correct_row(row, sample_rate, key, scale, correction_speed, mix))
    return copy_audio(audio, _from_numpy(wet_np, waveform))


def varispeed_player(audio: dict, speed_ratio: float, output_gain_db: float) -> dict:
    return resampler_classic(audio, speed_ratio, output_gain_db)
