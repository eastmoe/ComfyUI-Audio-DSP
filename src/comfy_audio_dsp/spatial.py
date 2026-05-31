from __future__ import annotations

import math
import os

import numpy as np
import torch
from scipy import signal

from .common import audio_waveform, butter_sos, clamp01, copy_audio, db_to_amp, mix_audio, resample_np, sos_filter_waveform
from .reverb import schroeder_reverb

SPATIAL_DECODER_MODES = ["stereo", "binaural"]
HOA_ORDERS = ["1", "2", "3"]


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _from_numpy(data: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(data.astype(np.float32, copy=False)).to(device=like.device, dtype=like.dtype)


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.mean(dim=1, keepdim=True)


def _fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    positions = np.arange(x.shape[-1], dtype=np.float32) - float(delay_samples)
    return np.interp(positions, np.arange(x.shape[-1], dtype=np.float32), x, left=0.0, right=0.0).astype(np.float32)


def _simple_binaural_np(mono: np.ndarray, sample_rate: int, azimuth_deg: float, elevation_deg: float, distance_m: float = 1.0) -> np.ndarray:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    ear_span_s = 0.00063
    itd = ear_span_s * math.sin(az) * max(0.15, math.cos(el))
    distance_gain = 1.0 / max(1.0, float(distance_m))
    right_focus = 0.5 + 0.5 * math.sin(az)
    left_gain = distance_gain * (1.0 - 0.35 * right_focus)
    right_gain = distance_gain * (0.65 + 0.35 * right_focus)
    left_delay = max(0.0, itd) * sample_rate
    right_delay = max(0.0, -itd) * sample_rate
    left = _fractional_delay(mono, left_delay) * left_gain
    right = _fractional_delay(mono, right_delay) * right_gain
    if abs(float(elevation_deg)) > 1.0:
        cutoff = max(1200.0, min(sample_rate * 0.45, 9000.0 - abs(float(elevation_deg)) * 45.0))
        sos = signal.butter(1, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        if elevation_deg > 0.0:
            left = signal.sosfilt(sos, left).astype(np.float32)
        else:
            right = signal.sosfilt(sos, right).astype(np.float32)
    return np.stack([left, right], axis=0).astype(np.float32)


def binaural_panner(audio: dict, azimuth_deg: float, elevation_deg: float, distance_m: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    mono = _to_numpy(_mono(waveform))
    wet = np.empty((mono.shape[0], 2, mono.shape[-1]), dtype=np.float32)
    for b in range(mono.shape[0]):
        wet[b] = _simple_binaural_np(mono[b, 0], sample_rate, azimuth_deg, elevation_deg, distance_m)
    wet_tensor = _from_numpy(wet, waveform)
    dry = _mono(waveform).repeat(1, 2, 1)
    return copy_audio(audio, mix_audio(dry, wet_tensor, mix))


def _read_sofa_ir(path: str, sample_rate: int, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Comfy-Audio-DSP: SOFA file not found: {path}")
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("Comfy-Audio-DSP: HRTF SOFA loading needs h5py to be available in the ComfyUI environment.") from exc

    with h5py.File(path, "r") as handle:
        ir = np.asarray(handle["Data.IR"], dtype=np.float32)
        positions = np.asarray(handle["SourcePosition"], dtype=np.float32)
        sofa_rate = int(np.asarray(handle["Data.SamplingRate"]).reshape(-1)[0])

    if ir.ndim != 3:
        raise ValueError("Comfy-Audio-DSP: expected SOFA Data.IR with shape [M, R, N].")
    if ir.shape[1] != 2 and ir.shape[2] == 2:
        ir = np.transpose(ir, (0, 2, 1))
    az = positions[:, 0]
    el = positions[:, 1]
    score = np.abs(((az - float(azimuth_deg) + 180.0) % 360.0) - 180.0) + np.abs(el - float(elevation_deg))
    selected = ir[int(np.argmin(score))]
    if selected.shape[0] != 2:
        selected = selected[:2] if selected.shape[0] > 2 else np.repeat(selected[:1], 2, axis=0)
    return resample_np(selected, sofa_rate, sample_rate)


def hrtf_convolution(audio: dict, sofa_path: str, azimuth_deg: float, elevation_deg: float, normalize_ir: bool, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    mono = _to_numpy(_mono(waveform))
    ir = _read_sofa_ir(sofa_path, sample_rate, azimuth_deg, elevation_deg)
    if bool(normalize_ir):
        ir = ir / max(float(np.max(np.abs(ir))), 1.0e-6)
    wet = np.zeros((mono.shape[0], 2, mono.shape[-1]), dtype=np.float32)
    for b in range(mono.shape[0]):
        for ear in range(2):
            wet[b, ear] = signal.fftconvolve(mono[b, 0], ir[ear], mode="full")[: mono.shape[-1]].astype(np.float32)
    wet_tensor = _from_numpy(wet, waveform)
    dry = _mono(waveform).repeat(1, 2, 1)
    return copy_audio(audio, mix_audio(dry, wet_tensor, mix))


def ambisonics_encoder(audio: dict, azimuth_deg: float, elevation_deg: float, gain_db: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    mono = _mono(waveform) * float(db_to_amp(gain_db))
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    w = mono / math.sqrt(2.0)
    x = mono * math.cos(az) * math.cos(el)
    y = mono * math.sin(az) * math.cos(el)
    z = mono * math.sin(el)
    return copy_audio(audio, torch.cat([w, x, y, z], dim=1))


def _first_four(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if waveform.shape[1] < 4:
        waveform = torch.cat([waveform, torch.zeros(waveform.shape[0], 4 - waveform.shape[1], waveform.shape[-1], device=waveform.device, dtype=waveform.dtype)], dim=1)
    return waveform[:, :1], waveform[:, 1:2], waveform[:, 2:3], waveform[:, 3:4]


def ambisonics_decoder(audio: dict, mode: str, width: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    w, x, y, _z = _first_four(waveform)
    width = float(width)
    left = w / math.sqrt(2.0) + 0.5 * x + 0.5 * y * width
    right = w / math.sqrt(2.0) + 0.5 * x - 0.5 * y * width
    stereo = torch.cat([left, right], dim=1)
    if mode == "binaural":
        virtual_left = {"waveform": left, "sample_rate": sample_rate}
        virtual_right = {"waveform": right, "sample_rate": sample_rate}
        left_ear = binaural_panner(virtual_left, -30.0, 0.0, 1.0, 1.0)["waveform"]
        right_ear = binaural_panner(virtual_right, 30.0, 0.0, 1.0, 1.0)["waveform"]
        stereo = (left_ear + right_ear) * 0.5
    return copy_audio(audio, stereo)


def ambisonics_rotator(audio: dict, yaw_deg: float, pitch_deg: float, roll_deg: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    w, x, y, z = _first_four(waveform)
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z
    x2 = x1 * math.cos(pitch) + z1 * math.sin(pitch)
    z2 = -x1 * math.sin(pitch) + z1 * math.cos(pitch)
    y2 = y1
    y3 = y2 * math.cos(roll) - z2 * math.sin(roll)
    z3 = y2 * math.sin(roll) + z2 * math.cos(roll)
    return copy_audio(audio, torch.cat([w, x2, y3, z3], dim=1))


def distance_simulator(
    audio: dict,
    distance_m: float,
    air_absorption: float,
    room_mix: float,
    dry_gain_db: float,
    reverb_time_s: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    distance_m = max(0.0, float(distance_m))
    gain = 1.0 / (1.0 + distance_m)
    cutoff = max(700.0, min(sample_rate * 0.45, 20000.0 / (1.0 + distance_m * clamp01(air_absorption) * 0.12)))
    filtered = sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass", cutoff, order=2), zero_phase=False)
    dry = filtered * gain * float(db_to_amp(dry_gain_db))
    room = clamp01(room_mix) * clamp01(distance_m / 20.0)
    if room <= 0.0:
        return copy_audio(audio, dry)
    wet = schroeder_reverb({"waveform": filtered * gain, "sample_rate": sample_rate}, distance_m / 343.0 * 1000.0, reverb_time_s, 0.65, 120.0, cutoff, 1.0)["waveform"]
    return copy_audio(audio, dry * (1.0 - room) + wet * room)


def doppler_effect(audio: dict, start_distance_m: float, end_distance_m: float, source_speed_m_s: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    length = data.shape[-1]
    duration = max(length / float(sample_rate), 1.0e-6)
    natural_velocity = (float(end_distance_m) - float(start_distance_m)) / duration
    velocity = natural_velocity + float(source_speed_m_s)
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    distances = np.linspace(float(start_distance_m), float(end_distance_m), length, dtype=np.float32)
    emission_time = t - distances / 343.0
    emission_time -= float(np.min(emission_time))
    pitch_ratio = 343.0 / max(20.0, 343.0 + velocity)
    positions = emission_time * sample_rate * pitch_ratio
    dry_gain = 1.0 / np.maximum(1.0, 1.0 + distances)
    wet = np.zeros_like(data, dtype=np.float32)
    grid = np.arange(length, dtype=np.float32)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            wet[b, c] = np.interp(positions, grid, data[b, c], left=0.0, right=0.0).astype(np.float32) * dry_gain
    wet_tensor = _from_numpy(wet, waveform)
    return copy_audio(audio, mix_audio(waveform, wet_tensor, mix))


def vbap_panner(audio: dict, azimuth_deg: float, speaker_angles_deg: str, spread: float, normalize: bool) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    mono = _mono(waveform)
    angles = []
    for item in str(speaker_angles_deg).replace(";", ",").split(","):
        item = item.strip()
        if item:
            angles.append(float(item))
    if len(angles) < 2:
        angles = [-30.0, 30.0]
    target = ((float(azimuth_deg) + 180.0) % 360.0) - 180.0
    speakers = [((angle + 180.0) % 360.0) - 180.0 for angle in angles]
    order = np.argsort(speakers)
    speakers_sorted = [speakers[i] for i in order]
    gains_sorted = np.zeros(len(speakers_sorted), dtype=np.float32)
    extended = speakers_sorted + [speakers_sorted[0] + 360.0]
    target_ext = target
    if target_ext < speakers_sorted[0]:
        target_ext += 360.0
    pair = 0
    for index in range(len(speakers_sorted)):
        if extended[index] <= target_ext <= extended[index + 1]:
            pair = index
            break
    left_angle = extended[pair]
    right_angle = extended[pair + 1]
    frac = 0.0 if right_angle == left_angle else (target_ext - left_angle) / (right_angle - left_angle)
    gains_sorted[pair % len(speakers_sorted)] = math.cos(frac * math.pi * 0.5)
    gains_sorted[(pair + 1) % len(speakers_sorted)] = math.sin(frac * math.pi * 0.5)
    if spread > 0.0:
        for index, angle in enumerate(speakers_sorted):
            dist = abs(((angle - target + 180.0) % 360.0) - 180.0)
            gains_sorted[index] += max(0.0, 1.0 - dist / max(float(spread), 1.0)) * 0.5
    if bool(normalize):
        gains_sorted /= max(float(np.sqrt(np.sum(gains_sorted * gains_sorted))), 1.0e-6)
    gains = np.zeros_like(gains_sorted)
    for sorted_index, original_index in enumerate(order):
        gains[original_index] = gains_sorted[sorted_index]
    out = torch.cat([mono * float(gain) for gain in gains], dim=1)
    return copy_audio(audio, out)


def higher_order_ambisonics_encoder(audio: dict, order: str, azimuth_deg: float, elevation_deg: float, gain_db: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    mono = _mono(waveform) * float(db_to_amp(gain_db))
    order_i = max(1, min(int(order), 3))
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    channels = [mono / math.sqrt(2.0)]
    for n in range(1, order_i + 1):
        channels.append(mono * math.cos(n * az) * math.cos(el) ** n)
        channels.append(mono * math.sin(n * az) * math.cos(el) ** n)
        channels.append(mono * math.sin(el) ** n)
        for _m in range(max(0, 2 * n - 2)):
            channels.append(torch.zeros_like(mono))
    return copy_audio(audio, torch.cat(channels[: (order_i + 1) ** 2], dim=1))


def _pad_hoa(waveform: torch.Tensor, order_i: int) -> torch.Tensor:
    channels = (order_i + 1) ** 2
    if waveform.shape[1] < channels:
        waveform = torch.cat([waveform, torch.zeros(waveform.shape[0], channels - waveform.shape[1], waveform.shape[-1], device=waveform.device, dtype=waveform.dtype)], dim=1)
    return waveform[:, :channels]


def higher_order_ambisonics_decoder(audio: dict, order: str, mode: str, speaker_angles_deg: str, width: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    order_i = max(1, min(int(order), 3))
    hoa = _pad_hoa(waveform, order_i)
    if mode in {"stereo", "binaural"}:
        angles = [-30.0, 30.0]
    else:
        angles = [float(item.strip()) for item in speaker_angles_deg.replace(";", ",").split(",") if item.strip()] or [-30.0, 30.0]
    outs = []
    for angle in angles:
        value = hoa[:, :1] / math.sqrt(2.0)
        index = 1
        az = math.radians(angle)
        for n in range(1, order_i + 1):
            value = value + (hoa[:, index : index + 1] * math.cos(n * az) + hoa[:, index + 1 : index + 2] * math.sin(n * az)) * float(width) / n
            index += 2 * n + 1
        outs.append(value)
    return copy_audio(audio, torch.cat(outs, dim=1))


def higher_order_ambisonics_rotator(audio: dict, order: str, yaw_deg: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    order_i = max(1, min(int(order), 3))
    hoa = _pad_hoa(waveform, order_i)
    out = hoa.clone()
    yaw = math.radians(float(yaw_deg))
    index = 1
    for n in range(1, order_i + 1):
        cos_ch = hoa[:, index : index + 1]
        sin_ch = hoa[:, index + 1 : index + 2]
        angle = n * yaw
        out[:, index : index + 1] = cos_ch * math.cos(angle) - sin_ch * math.sin(angle)
        out[:, index + 1 : index + 2] = cos_ch * math.sin(angle) + sin_ch * math.cos(angle)
        index += 2 * n + 1
    return copy_audio(audio, out)
