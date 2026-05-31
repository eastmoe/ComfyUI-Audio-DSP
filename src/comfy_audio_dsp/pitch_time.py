from __future__ import annotations

import math

import numpy as np
import torch
from scipy import signal

from .common import AUDIO_EPS, audio_waveform, clamp01, copy_audio, db_to_amp, flatten_channels, mix_audio, restore_channels

PITCH_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_SCALES = ["chromatic", "major", "minor", "major_pentatonic", "minor_pentatonic"]
FORMANT_MODES = ["up", "down"]
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


def granular_processor(
    audio: dict,
    grain_ms: float,
    overlap: float,
    pitch_semitones: float,
    position_jitter_ms: float,
    time_scatter: float,
    reverse_probability: float,
    seed: int,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    rng = np.random.default_rng(int(seed))
    grain = max(8, int(round(float(grain_ms) * sample_rate / 1000.0)))
    hop = max(1, int(round(grain / max(float(overlap), 1.0))))
    jitter = max(0, int(round(float(position_jitter_ms) * sample_rate / 1000.0)))
    pitch_ratio = 2.0 ** (float(pitch_semitones) / 12.0)
    window = np.hanning(grain).astype(np.float32)
    wet = np.zeros_like(data, dtype=np.float32)
    weight = np.zeros_like(data, dtype=np.float32)
    starts = list(range(0, data.shape[-1], hop))
    scatter = max(0.0, min(float(time_scatter), 1.0))
    if scatter > 0.0:
        shuffled = starts.copy()
        rng.shuffle(shuffled)
        starts = [int(round((1.0 - scatter) * a + scatter * b)) for a, b in zip(starts, shuffled, strict=False)]
    out_positions = range(0, data.shape[-1], hop)
    for out_start, source_base in zip(out_positions, starts, strict=False):
        source_start = int(source_base + rng.integers(-jitter, jitter + 1)) if jitter > 0 else int(source_base)
        source_start = max(0, min(data.shape[-1] - 1, source_start))
        source_end = min(data.shape[-1], source_start + grain)
        if source_end <= source_start:
            continue
        out_end = min(data.shape[-1], out_start + grain)
        size = out_end - out_start
        if size <= 0:
            continue
        chunk = data[..., source_start:source_end]
        if chunk.shape[-1] < grain:
            chunk = np.pad(chunk, [(0, 0), (0, 0), (0, grain - chunk.shape[-1])])
        if abs(pitch_ratio - 1.0) > 1.0e-4:
            shifted = np.empty_like(chunk)
            for batch in range(chunk.shape[0]):
                for channel in range(chunk.shape[1]):
                    shifted[batch, channel] = _fit_length(_classic_resample_row(chunk[batch, channel], pitch_ratio), grain)
            chunk = shifted
        if rng.random() < max(0.0, min(float(reverse_probability), 1.0)):
            chunk = chunk[..., ::-1]
        env = window[:size]
        wet[..., out_start:out_end] += chunk[..., :size] * env
        weight[..., out_start:out_end] += env
    wet = wet / np.maximum(weight, 1.0e-5)
    wet_tensor = _from_numpy(wet, waveform)
    return copy_audio(audio, mix_audio(waveform, wet_tensor, mix))


def formant_shifter(audio: dict, shift_ratio: float, fft_size: int, hop_size: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy().astype(np.float32, copy=False)
    fft_size = max(128, int(fft_size))
    hop_size = max(1, int(hop_size))
    ratio = max(0.25, min(float(shift_ratio), 4.0))
    out = np.zeros_like(data, dtype=np.float32)
    for row in range(data.shape[0]):
        _, _, stft = signal.stft(data[row], fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), boundary="zeros", padded=True)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        bins = np.arange(mag.shape[0], dtype=np.float32)
        warped = np.empty_like(mag)
        for frame in range(mag.shape[1]):
            warped[:, frame] = np.interp(bins / ratio, bins, mag[:, frame], left=mag[0, frame], right=0.0)
        _, y = signal.istft(warped * phase, fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), input_onesided=True)
        if y.shape[-1] < data.shape[-1]:
            y = np.pad(y, (0, data.shape[-1] - y.shape[-1]))
        out[row] = y[: data.shape[-1]]
    wet = restore_channels(torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype), shape)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def polyphonic_pitch_correction(audio: dict, key: str, scale: str, correction_amount: float, attenuation_db: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sample_rate)
    key_offset = PITCH_KEYS.index(key) if key in PITCH_KEYS else 0
    intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS["chromatic"])
    response = np.ones_like(freqs, dtype=np.float32)
    for index, freq in enumerate(freqs):
        if freq < 20.0:
            continue
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        chroma = (round(midi) - key_offset) % 12
        if chroma not in intervals:
            distance = min(abs(chroma - interval) % 12 for interval in intervals)
            response[index] = float(db_to_amp(-abs(float(attenuation_db)) * max(0.0, min(float(correction_amount), 1.0)) / max(distance, 1)))
    wet_np = np.fft.irfft(np.fft.rfft(data, axis=-1) * response[None, None, :], n=data.shape[-1], axis=-1).astype(np.float32)
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet_np, waveform), mix))
