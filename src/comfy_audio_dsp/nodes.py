from __future__ import annotations

from . import dsp

CATEGORY_DYNAMICS = "eastmoe/Comfy-Audio-DSP/Dynamics"


def _audio_input() -> tuple[str]:
    return ("AUDIO",)


class ComfyAudioDSPCompressor:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "threshold_db": ("FLOAT", {"default": -18.0, "min": -80.0, "max": 0.0, "step": 0.1}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 50.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "knee_db": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 48.0, "step": 0.1}),
                "makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 48.0, "step": 0.1}),
                "detector": (["rms", "peak"], {"default": "rms"}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db, detector, mix):
        return (dsp.compressor(audio, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db, detector, mix),)


class ComfyAudioDSPLimiter:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "threshold_db": ("FLOAT", {"default": -1.0, "min": -36.0, "max": 0.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 80.0, "min": 1.0, "max": 2000.0, "step": 1.0}),
                "lookahead_ms": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    def process(self, audio, threshold_db, release_ms, lookahead_ms):
        return (dsp.limiter(audio, threshold_db, release_ms, lookahead_ms),)


class ComfyAudioDSPNoiseGate:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -100.0, "max": 0.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "hold_ms": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "release_ms": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "range_db": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 120.0, "step": 0.5}),
            }
        }

    def process(self, audio, threshold_db, attack_ms, hold_ms, release_ms, range_db):
        return (dsp.noise_gate(audio, threshold_db, attack_ms, hold_ms, release_ms, range_db),)


class ComfyAudioDSPExpander:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "threshold_db": ("FLOAT", {"default": -40.0, "min": -100.0, "max": 0.0, "step": 0.1}),
                "ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "range_db": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 120.0, "step": 0.5}),
            }
        }

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, range_db):
        return (dsp.expander(audio, threshold_db, ratio, attack_ms, release_ms, range_db),)


class ComfyAudioDSPTransientShaper:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "attack_gain_db": ("FLOAT", {"default": 3.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "sustain_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "fast_ms": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 50.0, "step": 0.1}),
                "slow_ms": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 500.0, "step": 1.0}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self, audio, attack_gain_db, sustain_gain_db, fast_ms, slow_ms, mix):
        return (dsp.transient_shaper(audio, attack_gain_db, sustain_gain_db, fast_ms, slow_ms, mix),)


class ComfyAudioDSPDeEsser:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "threshold_db": ("FLOAT", {"default": -30.0, "min": -80.0, "max": 0.0, "step": 0.1}),
                "ratio": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 80.0, "min": 1.0, "max": 2000.0, "step": 1.0}),
                "frequency_low_hz": ("FLOAT", {"default": 4000.0, "min": 500.0, "max": 20000.0, "step": 10.0}),
                "frequency_high_hz": ("FLOAT", {"default": 10000.0, "min": 1000.0, "max": 22000.0, "step": 10.0}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, frequency_low_hz, frequency_high_hz, amount):
        return (dsp.de_esser(audio, threshold_db, ratio, attack_ms, release_ms, frequency_low_hz, frequency_high_hz, amount),)


class ComfyAudioDSPMultiBandCompressor:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "bands": (["3", "4"], {"default": "4"}),
                "crossover_low_hz": ("FLOAT", {"default": 160.0, "min": 40.0, "max": 2000.0, "step": 1.0}),
                "crossover_mid_hz": ("FLOAT", {"default": 1200.0, "min": 100.0, "max": 8000.0, "step": 1.0}),
                "crossover_high_hz": ("FLOAT", {"default": 6000.0, "min": 1000.0, "max": 20000.0, "step": 1.0}),
                "low_threshold_db": ("FLOAT", {"default": -20.0, "min": -80.0, "max": 0.0, "step": 0.1}),
                "low_ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "low_makeup_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "low_mid_threshold_db": ("FLOAT", {"default": -18.0, "min": -80.0, "max": 0.0, "step": 0.1}),
                "low_mid_ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "low_mid_makeup_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "high_mid_threshold_db": ("FLOAT", {"default": -16.0, "min": -80.0, "max": 0.0, "step": 0.1}),
                "high_mid_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "high_mid_makeup_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "high_threshold_db": ("FLOAT", {"default": -18.0, "min": -80.0, "max": 0.0, "step": 0.1}),
                "high_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "high_makeup_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "attack_ms": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "knee_db": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 48.0, "step": 0.1}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self, audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz, low_threshold_db, low_ratio, low_makeup_db, low_mid_threshold_db, low_mid_ratio, low_mid_makeup_db, high_mid_threshold_db, high_mid_ratio, high_mid_makeup_db, high_threshold_db, high_ratio, high_makeup_db, attack_ms, release_ms, knee_db, mix):
        return (dsp.multiband_compressor(audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz, low_threshold_db, low_ratio, low_makeup_db, low_mid_threshold_db, low_mid_ratio, low_mid_makeup_db, high_mid_threshold_db, high_mid_ratio, high_mid_makeup_db, high_threshold_db, high_ratio, high_makeup_db, attack_ms, release_ms, knee_db, mix),)


class ComfyAudioDSPAutoGainLeveler:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "mode": (["RMS", "Peak"], {"default": "RMS"}),
                "target_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 0.0, "step": 0.1}),
                "window_ms": ("FLOAT", {"default": 500.0, "min": 10.0, "max": 10000.0, "step": 10.0}),
                "attack_ms": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 5000.0, "step": 1.0}),
                "release_ms": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
                "min_gain_db": ("FLOAT", {"default": -24.0, "min": -60.0, "max": 0.0, "step": 0.1}),
                "max_gain_db": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            }
        }

    def process(self, audio, mode, target_db, window_ms, attack_ms, release_ms, min_gain_db, max_gain_db):
        return (dsp.auto_gain_leveler(audio, mode, target_db, window_ms, attack_ms, release_ms, min_gain_db, max_gain_db),)


class ComfyAudioDSPLoudnessNormalizer:
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": _audio_input(),
                "target_lufs": ("FLOAT", {"default": -16.0, "min": -40.0, "max": 0.0, "step": 0.1}),
                "measurement": (["integrated", "short_term", "momentary"], {"default": "integrated"}),
                "min_gain_db": ("FLOAT", {"default": -24.0, "min": -60.0, "max": 0.0, "step": 0.1}),
                "max_gain_db": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "true_peak_ceiling_db": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1}),
            }
        }

    def process(self, audio, target_lufs, measurement, min_gain_db, max_gain_db, true_peak_ceiling_db):
        return (dsp.loudness_normalizer(audio, target_lufs, measurement, min_gain_db, max_gain_db, true_peak_ceiling_db),)


NODE_CLASS_MAPPINGS = {
    "ComfyAudioDSPCompressor": ComfyAudioDSPCompressor,
    "ComfyAudioDSPLimiter": ComfyAudioDSPLimiter,
    "ComfyAudioDSPNoiseGate": ComfyAudioDSPNoiseGate,
    "ComfyAudioDSPExpander": ComfyAudioDSPExpander,
    "ComfyAudioDSPTransientShaper": ComfyAudioDSPTransientShaper,
    "ComfyAudioDSPDeEsser": ComfyAudioDSPDeEsser,
    "ComfyAudioDSPMultiBandCompressor": ComfyAudioDSPMultiBandCompressor,
    "ComfyAudioDSPAutoGainLeveler": ComfyAudioDSPAutoGainLeveler,
    "ComfyAudioDSPLoudnessNormalizer": ComfyAudioDSPLoudnessNormalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyAudioDSPCompressor": "Compressor",
    "ComfyAudioDSPLimiter": "Limiter",
    "ComfyAudioDSPNoiseGate": "Noise Gate",
    "ComfyAudioDSPExpander": "Expander",
    "ComfyAudioDSPTransientShaper": "Transient Shaper",
    "ComfyAudioDSPDeEsser": "De-Esser",
    "ComfyAudioDSPMultiBandCompressor": "Multi-band Compressor",
    "ComfyAudioDSPAutoGainLeveler": "Auto Gain / Leveler",
    "ComfyAudioDSPLoudnessNormalizer": "Loudness Normalizer (LUFS)",
}
