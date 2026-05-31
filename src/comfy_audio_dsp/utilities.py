from __future__ import annotations

import ast
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from .common import AUDIO_EPS, amp_to_db, audio_waveform, butter_sos, copy_audio, db_to_amp, meter_envelope, resample_np, sos_filter_waveform
from .dynamics import _integrated_lufs, _loudness_weighted

FADE_CURVES = ["linear", "exponential", "s_curve"]
NORMALIZE_MODES = ["peak", "rms", "lufs"]
FORMAT_MODES = ["mono_mix", "mono_left", "mono_right", "stereo_duplicate"]


def gain_trim(audio: dict, gain_db: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, waveform * float(db_to_amp(gain_db)))


def phase_inverter(audio: dict, invert_left: bool, invert_right: bool) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    out = waveform.clone()
    if out.shape[1] == 1:
        if bool(invert_left) or bool(invert_right):
            out = -out
    else:
        if bool(invert_left):
            out[:, :1] = -out[:, :1]
        if bool(invert_right):
            out[:, 1:2] = -out[:, 1:2]
    return copy_audio(audio, out)


def dc_offset_remover(audio: dict, highpass_hz: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if highpass_hz <= 0.0:
        return copy_audio(audio, waveform - waveform.mean(dim=-1, keepdim=True))
    return copy_audio(audio, sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", highpass_hz, order=2), zero_phase=False))


def _fade_curve(length: int, curve: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, max(1, int(length)), device=device, dtype=dtype)
    if curve == "exponential":
        return x * x
    if curve == "s_curve":
        return x * x * (3.0 - 2.0 * x)
    return x


def fade_in_out(audio: dict, fade_in_ms: float, fade_out_ms: float, curve: str) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    env = torch.ones(waveform.shape[-1], device=waveform.device, dtype=waveform.dtype)
    fade_in = min(waveform.shape[-1], max(0, int(round(float(fade_in_ms) * sample_rate / 1000.0))))
    fade_out = min(waveform.shape[-1], max(0, int(round(float(fade_out_ms) * sample_rate / 1000.0))))
    if fade_in > 0:
        env[:fade_in] *= _fade_curve(fade_in, curve, waveform.device, waveform.dtype)
    if fade_out > 0:
        env[-fade_out:] *= torch.flip(_fade_curve(fade_out, curve, waveform.device, waveform.dtype), dims=(0,))
    return copy_audio(audio, waveform * env.view(1, 1, -1))


def audio_trim_crop(audio: dict, start_s: float, end_s: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    start = max(0, int(round(float(start_s) * sample_rate)))
    end = waveform.shape[-1] if end_s <= 0.0 else min(waveform.shape[-1], int(round(float(end_s) * sample_rate)))
    end = max(start + 1, end)
    return copy_audio(audio, waveform[..., start:end])


def silence_trimmer(audio: dict, threshold_db: float, padding_ms: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.abs().amax(dim=1).squeeze(0)
    threshold = float(db_to_amp(threshold_db))
    active = torch.nonzero(mono > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return copy_audio(audio, waveform[..., :1] * 0.0)
    padding = max(0, int(round(float(padding_ms) * sample_rate / 1000.0)))
    start = max(0, int(active[0].item()) - padding)
    end = min(waveform.shape[-1], int(active[-1].item()) + padding + 1)
    return copy_audio(audio, waveform[..., start:end])


def normalize_audio(audio: dict, mode: str, target_db: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if mode == "rms":
        current = amp_to_db(torch.sqrt(torch.mean(waveform * waveform, dim=(1, 2), keepdim=True) + AUDIO_EPS))
    elif mode == "lufs":
        current = _integrated_lufs(_loudness_weighted(waveform, sample_rate), sample_rate).view(-1, 1, 1)
    else:
        current = amp_to_db(torch.amax(torch.abs(waveform), dim=(1, 2), keepdim=True) + AUDIO_EPS)
    gain_db = float(target_db) - current
    return copy_audio(audio, waveform * db_to_amp(gain_db))


def resample_change_sample_rate(audio: dict, target_sample_rate: int) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = waveform.detach().cpu().numpy().astype("float32", copy=False)
    out = resample_np(data, sample_rate, int(target_sample_rate))
    return {"waveform": torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype), "sample_rate": int(target_sample_rate)}


def format_converter(audio: dict, mode: str) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    if mode == "stereo_duplicate":
        out = waveform.repeat(1, 2, 1)[:, :2] if waveform.shape[1] == 1 else waveform[:, :2]
    elif mode == "mono_left":
        out = waveform[:, :1]
    elif mode == "mono_right":
        out = waveform[:, 1:2] if waveform.shape[1] > 1 else waveform[:, :1]
    else:
        out = waveform.mean(dim=1, keepdim=True)
    return copy_audio(audio, out)


def audio_info(audio: dict) -> tuple[dict, int, float, int, int, str]:
    waveform, sample_rate = audio_waveform(audio)
    duration = waveform.shape[-1] / max(float(sample_rate), 1.0)
    text = json.dumps({"sample_rate": sample_rate, "duration_s": duration, "channels": waveform.shape[1], "samples": waveform.shape[-1]}, ensure_ascii=False)
    return copy_audio(audio, waveform), int(sample_rate), float(duration), int(waveform.shape[1]), int(waveform.shape[-1]), text


def delay_compensation(audio: dict, delay_samples: int, delay_ms: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    total = max(0, int(delay_samples) + int(round(float(delay_ms) * sample_rate / 1000.0)))
    return copy_audio(audio, F.pad(waveform, (total, 0)))


def loop_duplicator(audio: dict, loops: int, target_duration_s: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if target_duration_s > 0.0:
        target = max(1, int(round(float(target_duration_s) * sample_rate)))
        repeats = max(1, int(math.ceil(target / waveform.shape[-1])))
        out = waveform.repeat(1, 1, repeats)[..., :target]
    else:
        out = waveform.repeat(1, 1, max(1, int(loops)))
    return copy_audio(audio, out)


def reverse_audio(audio: dict) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, torch.flip(waveform, dims=(-1,)))


def envelope_follower_output(audio: dict, attack_ms: float, release_ms: float, mode: str, normalize: bool, points: int) -> tuple[dict, float, float, str, dict]:
    waveform, sample_rate = audio_waveform(audio)
    source = waveform.abs().amax(dim=1, keepdim=True) if mode == "peak" else torch.sqrt(torch.mean(waveform * waveform, dim=1, keepdim=True) + AUDIO_EPS)
    env = meter_envelope(source.reshape(-1, source.shape[-1]), sample_rate, attack_ms, release_ms, mode="peak").reshape(source.shape)
    if bool(normalize):
        env = env / torch.clamp(env.amax(dim=-1, keepdim=True), min=AUDIO_EPS)
    current = float(env[..., -1].mean().item())
    average = float(env.mean().item())
    count = max(2, int(points))
    idx = torch.linspace(0, env.shape[-1] - 1, min(count, env.shape[-1]), device=env.device).long()
    values = env[0, 0, idx].detach().cpu().tolist()
    return copy_audio(audio, waveform), current, average, json.dumps(values, ensure_ascii=False), copy_audio(audio, env.repeat(1, waveform.shape[1], 1))


def declick_decrackle(audio: dict, threshold: float, window_samples: int, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    data = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    window = max(3, int(window_samples) | 1)
    out = np.empty_like(data, dtype=np.float32)
    for batch in range(data.shape[0]):
        for channel in range(data.shape[1]):
            x = data[batch, channel]
            median = signal.medfilt(x, kernel_size=window)
            residual = x - median
            mad = np.median(np.abs(residual - np.median(residual))) + 1.0e-8
            mask = np.abs(residual) > max(float(threshold), 1.0) * 1.4826 * mad
            repaired = x.copy()
            repaired[mask] = median[mask]
            out[batch, channel] = repaired
    wet = torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype)
    return copy_audio(audio, waveform.lerp(wet, max(0.0, min(float(mix), 1.0))))


def phase_rotator_allpass(audio: dict, frequency_hz: float, q: float, stages: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    omega = 2.0 * math.pi * max(20.0, min(float(frequency_hz), sample_rate * 0.45)) / sample_rate
    alpha = math.sin(omega) / (2.0 * max(float(q), 0.05))
    cos_omega = math.cos(omega)
    b0 = (1.0 - alpha) / (1.0 + alpha)
    b1 = -2.0 * cos_omega / (1.0 + alpha)
    b = np.array([b0, b1, 1.0], dtype=np.float64)
    a = np.array([1.0, b1, b0], dtype=np.float64)
    data = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    out = data.copy()
    for _ in range(max(1, int(stages))):
        for row in range(out.reshape(-1, out.shape[-1]).shape[0]):
            flat = out.reshape(-1, out.shape[-1])
            flat[row] = signal.lfilter(b, a, flat[row]).astype(np.float32)
    wet = torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype)
    return copy_audio(audio, waveform.lerp(wet, max(0.0, min(float(mix), 1.0))))


_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Call,
)


def _safe_eval_expression(expression: str, names: dict[str, torch.Tensor | float]) -> torch.Tensor:
    tree = ast.parse(expression, mode="eval")
    allowed_funcs = {
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "tanh": torch.tanh,
        "abs": torch.abs,
        "sqrt": torch.sqrt,
        "log": torch.log,
        "exp": torch.exp,
        "clamp": torch.clamp,
        "min": torch.minimum,
        "max": torch.maximum,
    }
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ValueError(f"Comfy-Audio-DSP: unsupported expression element {type(node).__name__}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("Comfy-Audio-DSP: formula only allows whitelisted math functions.")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in allowed_funcs:
            raise ValueError(f"Comfy-Audio-DSP: unknown formula name {node.id}.")
    return eval(compile(tree, "<comfy_audio_dsp_formula>", "eval"), {"__builtins__": {}, **allowed_funcs}, names)


def math_signal_mixer(audio_a: dict, audio_b: dict | None, audio_c: dict | None, audio_d: dict | None, expression: str, gain_db: float) -> dict:
    from .routing import _match_audio

    a, sample_rate = audio_waveform(audio_a)
    length = max([a.shape[-1]] + [audio_waveform(audio)[0].shape[-1] for audio in (audio_b, audio_c, audio_d) if audio is not None])
    channels = max([a.shape[1]] + [audio_waveform(audio)[0].shape[1] for audio in (audio_b, audio_c, audio_d) if audio is not None])
    A = _match_audio(audio_a, sample_rate, length, channels, a)
    zeros = torch.zeros_like(A)
    B = _match_audio(audio_b, sample_rate, length, channels, A) if audio_b is not None else zeros
    C = _match_audio(audio_c, sample_rate, length, channels, A) if audio_c is not None else zeros
    D = _match_audio(audio_d, sample_rate, length, channels, A) if audio_d is not None else zeros
    t = torch.arange(length, device=A.device, dtype=A.dtype).view(1, 1, -1) / float(sample_rate)
    names = {"A": A, "B": B, "C": C, "D": D, "t": t, "pi": math.pi}
    out = _safe_eval_expression(expression or "A", names)
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, device=A.device, dtype=A.dtype) + torch.zeros_like(A)
    out = out.to(device=A.device, dtype=A.dtype) * float(db_to_amp(gain_db))
    return copy_audio(audio_a, torch.clamp(out, min=-4.0, max=4.0))
