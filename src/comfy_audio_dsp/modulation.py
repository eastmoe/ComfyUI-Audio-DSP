from __future__ import annotations

import math

import numpy as np
import torch

from .common import audio_waveform, ba_filter_waveform, butter_sos, clamp01, copy_audio, db_to_amp, meter_envelope, mix_audio, sos_filter_waveform

WAVEFORMS = ["sine", "triangle", "square"]
MOD_SOURCE_WAVEFORMS = ["sine", "triangle", "square", "saw", "random"]
AUTO_FILTER_TYPES = ["low_pass", "high_pass", "band_pass"]


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _from_numpy(data: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(data.astype(np.float32, copy=False)).to(device=like.device, dtype=like.dtype)


def _lfo(length: int, sample_rate: int, rate_hz: float, waveform: str = "sine", phase: float = 0.0) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    phase_value = (t * max(float(rate_hz), 0.0) + phase / (2.0 * math.pi)) % 1.0
    if waveform == "triangle":
        return (4.0 * np.abs(phase_value - 0.5) - 1.0).astype(np.float32)
    if waveform == "square":
        return np.where(phase_value < 0.5, 1.0, -1.0).astype(np.float32)
    if waveform == "saw":
        return (phase_value * 2.0 - 1.0).astype(np.float32)
    if waveform == "random":
        steps = np.floor(phase_value * max(float(rate_hz), 1.0)).astype(np.int64)
        rng = np.random.default_rng(0)
        values = rng.uniform(-1.0, 1.0, int(np.max(steps)) + 2).astype(np.float32)
        return values[steps]
    return np.sin(2.0 * math.pi * phase_value).astype(np.float32)


def _control_audio(values: np.ndarray, sample_rate: int) -> dict:
    tensor = torch.from_numpy(values.astype(np.float32, copy=False)).view(1, 1, -1)
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _control_summary(values: np.ndarray, points: int) -> str:
    count = max(4, min(int(points), 1024))
    if values.shape[-1] > count:
        indices = np.linspace(0, values.shape[-1] - 1, count).astype(int)
        sampled = values[indices]
    else:
        sampled = values
    return "[" + ",".join(f"{float(v):.6g}" for v in sampled.tolist()) + "]"


def lfo_source(rate_hz: float, depth: float, offset: float, waveform_shape: str, duration_s: float, sample_rate: int, points: int) -> tuple[dict, float, str]:
    length = max(1, int(round(float(duration_s) * int(sample_rate))))
    values = float(offset) + clamp01(depth) * _lfo(length, int(sample_rate), rate_hz, waveform_shape)
    return _control_audio(values, int(sample_rate)), float(values[0]), _control_summary(values, points)


def adsr_envelope_generator(attack_ms: float, decay_ms: float, sustain_level: float, release_ms: float, gate_s: float, duration_s: float, sample_rate: int, points: int) -> tuple[dict, float, str]:
    sample_rate = int(sample_rate)
    length = max(1, int(round(float(duration_s) * sample_rate)))
    values = np.zeros(length, dtype=np.float32)
    attack = max(0, int(round(float(attack_ms) * sample_rate / 1000.0)))
    decay = max(0, int(round(float(decay_ms) * sample_rate / 1000.0)))
    release = max(0, int(round(float(release_ms) * sample_rate / 1000.0)))
    gate = min(length, max(0, int(round(float(gate_s) * sample_rate))))
    sustain = clamp01(sustain_level)
    cursor = 0
    if attack > 0:
        end = min(length, attack)
        values[:end] = np.linspace(0.0, 1.0, end, endpoint=False, dtype=np.float32)
        cursor = end
    elif length > 0:
        values[0] = 1.0
    if decay > 0 and cursor < gate:
        end = min(gate, cursor + decay)
        values[cursor:end] = np.linspace(1.0, sustain, end - cursor, endpoint=False, dtype=np.float32)
        cursor = end
    if cursor < gate:
        values[cursor:gate] = sustain
    start_level = float(values[gate - 1]) if gate > 0 else 0.0
    if release > 0 and gate < length:
        end = min(length, gate + release)
        values[gate:end] = np.linspace(start_level, 0.0, end - gate, endpoint=False, dtype=np.float32)
    return _control_audio(values, sample_rate), float(values[0]), _control_summary(values, points)


def sample_and_hold(rate_hz: float, smoothing: float, seed: int, duration_s: float, sample_rate: int, points: int) -> tuple[dict, float, str]:
    sample_rate = int(sample_rate)
    length = max(1, int(round(float(duration_s) * sample_rate)))
    step = max(1, int(round(sample_rate / max(float(rate_hz), 0.01))))
    rng = np.random.default_rng(int(seed))
    held = np.repeat(rng.uniform(0.0, 1.0, int(math.ceil(length / step)) + 1).astype(np.float32), step)[:length]
    if smoothing > 0.0:
        size = max(1, int(round(step * clamp01(smoothing))))
        kernel = np.hanning(size * 2 + 1).astype(np.float32)
        kernel /= max(float(np.sum(kernel)), 1.0e-8)
        held = np.convolve(held, kernel, mode="same").astype(np.float32)
    return _control_audio(held, sample_rate), float(held[0]), _control_summary(held, points)


def step_sequencer(sequence: str, bpm: float, step_value: str, glide: float, duration_s: float, sample_rate: int, points: int) -> tuple[dict, float, str]:
    sample_rate = int(sample_rate)
    length = max(1, int(round(float(duration_s) * sample_rate)))
    raw = [part.strip() for part in sequence.replace(";", ",").split(",") if part.strip()]
    steps = np.asarray([float(part) for part in raw] or [0.0], dtype=np.float32)
    divisions = {"1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25}
    beats = divisions.get(step_value, 0.25)
    step_samples = max(1, int(round(60.0 / max(float(bpm), 1.0) * beats * sample_rate)))
    values = np.repeat(steps, step_samples)
    values = np.resize(values, length).astype(np.float32)
    if glide > 0.0:
        size = max(1, int(round(step_samples * clamp01(glide))))
        kernel = np.hanning(size * 2 + 1).astype(np.float32)
        kernel /= max(float(np.sum(kernel)), 1.0e-8)
        values = np.convolve(values, kernel, mode="same").astype(np.float32)
    return _control_audio(values, sample_rate), float(values[0]), _control_summary(values, points)


def _fractional_delay(x: np.ndarray, delays: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    length = x.shape[-1]
    for n in range(length):
        index = n - float(delays[n])
        if 0.0 <= index < length - 1:
            left = int(math.floor(index))
            frac = index - left
            out[n] = x[left] * (1.0 - frac) + x[left + 1] * frac
    return out


def _mod_delay_channel(x: np.ndarray, sample_rate: int, base_ms: float, depth_ms: float, rate_hz: float, phase: float = 0.0, waveform: str = "sine") -> np.ndarray:
    lfo = _lfo(x.shape[-1], sample_rate, rate_hz, waveform, phase)
    delays = sample_rate * (float(base_ms) + float(depth_ms) * (0.5 + 0.5 * lfo)) / 1000.0
    return _fractional_delay(x, delays)


def chorus(audio: dict, voices: int, delay_ms: float, depth_ms: float, rate_hz: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.zeros_like(data, dtype=np.float32)
    voices = max(1, min(int(voices), 8))
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            channel = np.zeros(data.shape[-1], dtype=np.float32)
            last = np.zeros_like(channel)
            for voice in range(voices):
                phase = 2.0 * math.pi * voice / voices + c * math.pi * 0.5
                delayed = _mod_delay_channel(data[b, c] + last * float(feedback), sample_rate, delay_ms, depth_ms, rate_hz, phase)
                channel += delayed / voices
                last = delayed
            wet[b, c] = channel
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def flanger(audio: dict, base_delay_ms: float, depth_ms: float, rate_hz: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.empty_like(data)
    feedback = max(-0.95, min(float(feedback), 0.95))
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            delayed = _mod_delay_channel(data[b, c], sample_rate, base_delay_ms, depth_ms, rate_hz, c * math.pi)
            wet[b, c] = delayed + feedback * np.concatenate([np.zeros(1, dtype=np.float32), delayed[:-1]])
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def _first_order_allpass(x: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    x1 = 0.0
    y1 = 0.0
    for n in range(x.shape[-1]):
        a = float(coeff[n])
        y[n] = -a * x[n] + x1 + a * y1
        x1 = float(x[n])
        y1 = float(y[n])
    return y


def phaser(audio: dict, stages: int, rate_hz: float, depth: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.empty_like(data)
    stages = max(1, min(int(stages), 12))
    lfo_base = 0.5 + 0.5 * _lfo(data.shape[-1], sample_rate, rate_hz)
    coeff = 0.15 + clamp01(depth) * 0.75 * lfo_base
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            y = data[b, c].copy()
            for _ in range(stages):
                y = _first_order_allpass(y, coeff)
            wet[b, c] = y + float(feedback) * (y - data[b, c])
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def tremolo(audio: dict, rate_hz: float, depth: float, waveform_shape: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    lfo = torch.from_numpy(_lfo(waveform.shape[-1], sample_rate, rate_hz, waveform_shape)).to(device=waveform.device, dtype=waveform.dtype)
    gain = 1.0 - clamp01(depth) * (0.5 + 0.5 * lfo)
    wet = waveform * gain.view(1, 1, -1)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def vibrato(audio: dict, depth_ms: float, rate_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.empty_like(data)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            wet[b, c] = _mod_delay_channel(data[b, c], sample_rate, depth_ms, depth_ms, rate_hz, c * math.pi * 0.5)
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def rotary_speaker(audio: dict, low_rate_hz: float, high_rate_hz: float, depth: float, crossover_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    low = sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass", crossover_hz, order=2), zero_phase=False)
    high = sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", crossover_hz, order=2), zero_phase=False)
    length = waveform.shape[-1]
    low_lfo = torch.from_numpy(_lfo(length, sample_rate, low_rate_hz)).to(device=waveform.device, dtype=waveform.dtype)
    high_lfo = torch.from_numpy(_lfo(length, sample_rate, high_rate_hz, phase=math.pi * 0.5)).to(device=waveform.device, dtype=waveform.dtype)
    low_wet = low * (1.0 + 0.25 * clamp01(depth) * low_lfo.view(1, 1, -1))
    high_wet = high * (1.0 + 0.45 * clamp01(depth) * high_lfo.view(1, 1, -1))
    wet = low_wet + high_wet
    if wet.shape[1] >= 2:
        wet[:, 0] *= 1.0 - 0.25 * clamp01(depth) * high_lfo
        wet[:, 1] *= 1.0 + 0.25 * clamp01(depth) * high_lfo
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def ring_modulator(audio: dict, carrier_hz: float, depth: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    t = torch.arange(waveform.shape[-1], device=waveform.device, dtype=waveform.dtype) / float(sample_rate)
    carrier = torch.sin(2.0 * math.pi * float(carrier_hz) * t)
    wet = waveform * ((1.0 - clamp01(depth)) + clamp01(depth) * carrier.view(1, 1, -1))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def auto_panner(audio: dict, rate_hz: float, depth: float, waveform_shape: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    source = waveform if waveform.shape[1] >= 2 else waveform.repeat(1, 2, 1)
    lfo = torch.from_numpy(_lfo(source.shape[-1], sample_rate, rate_hz, waveform_shape)).to(device=source.device, dtype=source.dtype)
    pan = clamp01(depth) * lfo
    left = source[:, :1] * torch.sqrt(torch.clamp((1.0 - pan) * 0.5, min=0.0)).view(1, 1, -1)
    right = source[:, 1:2] * torch.sqrt(torch.clamp((1.0 + pan) * 0.5, min=0.0)).view(1, 1, -1)
    wet = torch.cat([left, right, source[:, 2:]], dim=1)
    return copy_audio(audio, mix_audio(source, wet, mix))


def uni_vibe(audio: dict, rate_hz: float, depth: float, chorus_mix: float, mix: float) -> dict:
    phased = phaser(audio, 4, rate_hz, depth, 0.15, 1.0)
    chorused = chorus(phased, 2, 7.0, 2.5 * clamp01(depth), rate_hz, 0.05, chorus_mix)
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, mix_audio(waveform, chorused["waveform"], mix))


def _match_modulator(modulator: dict, sample_rate: int, length: int, channels: int, like: torch.Tensor) -> torch.Tensor:
    waveform, mod_rate = audio_waveform(modulator)
    if mod_rate != sample_rate:
        raise ValueError("Comfy-Audio-DSP: vocoder carrier and modulator must have matching sample rates.")
    if waveform.shape[-1] < length:
        waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))
    elif waveform.shape[-1] > length:
        waveform = waveform[..., :length]
    if waveform.shape[1] < channels:
        waveform = waveform.repeat(1, int(math.ceil(channels / waveform.shape[1])), 1)[:, :channels]
    elif waveform.shape[1] > channels:
        waveform = waveform[:, :channels]
    return waveform.to(device=like.device, dtype=like.dtype)


def vocoder(
    carrier: dict,
    modulator: dict,
    bands: int,
    low_hz: float,
    high_hz: float,
    attack_ms: float,
    release_ms: float,
    modulator_gain_db: float,
    carrier_gain_db: float,
    mix: float,
) -> dict:
    carrier_waveform, sample_rate = audio_waveform(carrier)
    mod_waveform = _match_modulator(modulator, sample_rate, carrier_waveform.shape[-1], carrier_waveform.shape[1], carrier_waveform)
    bands = max(4, min(int(bands), 32))
    low = max(20.0, float(low_hz))
    high = max(low + 20.0, min(float(high_hz), sample_rate * 0.45))
    edges = np.geomspace(low, high, bands + 1)
    wet = torch.zeros_like(carrier_waveform)
    for index in range(bands):
        band_low = float(edges[index])
        band_high = float(edges[index + 1])
        carrier_band = sos_filter_waveform(carrier_waveform, butter_sos(sample_rate, "bandpass", (band_low, band_high), order=2), zero_phase=False)
        mod_band = sos_filter_waveform(mod_waveform, butter_sos(sample_rate, "bandpass", (band_low, band_high), order=2), zero_phase=False)
        env = meter_envelope(mod_band.reshape(-1, mod_band.shape[-1]), sample_rate, attack_ms, release_ms, mode="rms", frame_ms=2.0).reshape(mod_band.shape)
        band_gain = env * float(db_to_amp(modulator_gain_db))
        wet = wet + carrier_band * band_gain
    peak = wet.abs().amax(dim=(1, 2), keepdim=True)
    wet = wet / torch.clamp(peak, min=1.0)
    wet = wet * float(db_to_amp(carrier_gain_db))
    return copy_audio(carrier, mix_audio(carrier_waveform, wet, mix))


def barberpole_flanger(audio: dict, base_delay_ms: float, depth_ms: float, rate_hz: float, feedback: float, direction: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.zeros_like(data, dtype=np.float32)
    sign = -1.0 if direction == "down" else 1.0
    length = data.shape[-1]
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            acc = np.zeros(length, dtype=np.float32)
            for voice in range(4):
                phase = (np.arange(length, dtype=np.float32) * float(rate_hz) / sample_rate + voice / 4.0) % 1.0
                ramp = phase if sign > 0.0 else 1.0 - phase
                delays = sample_rate * (float(base_delay_ms) + float(depth_ms) * ramp) / 1000.0
                voice_out = _fractional_delay(data[b, c], delays)
                fade = np.sin(np.pi * phase) ** 2
                acc += voice_out * fade
            wet[b, c] = acc * 0.5 + feedback * np.concatenate([np.zeros(1, dtype=np.float32), acc[:-1]])
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def auto_filter(audio: dict, filter_type: str, base_cutoff_hz: float, depth_octaves: float, rate_hz: float, resonance_q: float, waveform_shape: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.zeros_like(data, dtype=np.float32)
    lfo = 0.5 + 0.5 * _lfo(data.shape[-1], sample_rate, rate_hz, waveform_shape)
    cutoff = float(base_cutoff_hz) * (2.0 ** ((lfo - 0.5) * 2.0 * float(depth_octaves)))
    cutoff = np.clip(cutoff, 20.0, sample_rate * 0.45)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            y = np.zeros(data.shape[-1], dtype=np.float32)
            z = 0.0
            for n in range(data.shape[-1]):
                alpha = 1.0 - math.exp(-2.0 * math.pi * float(cutoff[n]) / sample_rate)
                z = z + alpha * (float(data[b, c, n]) - z)
                if filter_type == "high_pass":
                    y[n] = data[b, c, n] - z
                elif filter_type == "band_pass":
                    y[n] = (data[b, c, n] - z) * min(max(float(resonance_q), 0.1), 10.0)
                else:
                    y[n] = z
            wet[b, c] = np.tanh(y) if filter_type == "band_pass" else y
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))
