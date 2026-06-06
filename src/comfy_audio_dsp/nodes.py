from __future__ import annotations

from . import dsp
from . import localization as loc
from .delay import NOTE_VALUES
from .equalizers import FILTER_TYPES, GRAPHIC_EQ_BANDS, HUM_BASE_MODES, LINEAR_PHASE_EQ_TYPES, SPECTRAL_SHAPER_MODES
from .generators import NOISE_TYPES, OSCILLATOR_WAVES, SWEEP_MODES, WAVETABLES
from .analysis import LOUDNESS_GRAPH_COLORS
from .modulation import AUTO_FILTER_TYPES, MOD_SOURCE_WAVEFORMS, STUTTER_DIVISIONS, WAVEFORMS
from .pitch_time import PITCH_KEYS, PITCH_SCALES
from .restoration import DENOISER_METHODS
from .spatial import HOA_ORDERS, SPATIAL_DECODER_MODES
from .stereo import MID_SIDE_EQ_FILTER_TYPES
from .utilities import DITHER_TYPES, FADE_CURVES, FORMAT_MODES, NORMALIZE_MODES

ROOT_CATEGORY = loc.category("root", "eastmoe/Comfy-Audio-DSP")
CATEGORY_DYNAMICS = f"{ROOT_CATEGORY}/{loc.category('dynamics', 'Dynamics')}"
CATEGORY_EQ = f"{ROOT_CATEGORY}/{loc.category('equalizers_filters', 'Equalizers & Filters')}"
CATEGORY_REVERB = f"{ROOT_CATEGORY}/{loc.category('reverb', 'Reverb')}"
CATEGORY_DELAY = f"{ROOT_CATEGORY}/{loc.category('delay_echo', 'Delay & Echo')}"
CATEGORY_MODULATION = f"{ROOT_CATEGORY}/{loc.category('modulation', 'Modulation')}"
CATEGORY_SATURATION = f"{ROOT_CATEGORY}/{loc.category('distortion_saturation', 'Distortion & Saturation')}"
CATEGORY_PITCH_TIME = f"{ROOT_CATEGORY}/{loc.category('pitch_time', 'Pitch & Time')}"
CATEGORY_STEREO = f"{ROOT_CATEGORY}/{loc.category('stereo_imaging', 'Stereo Imaging')}"
CATEGORY_SPATIAL = f"{ROOT_CATEGORY}/{loc.category('spatial_3d', 'Spatial & 3D')}"
CATEGORY_METERING = f"{ROOT_CATEGORY}/{loc.category('metering_analysis', 'Metering & Analysis')}"
CATEGORY_GENERATORS = f"{ROOT_CATEGORY}/{loc.category('signal_generators', 'Signal Generators')}"
CATEGORY_ROUTING = f"{ROOT_CATEGORY}/{loc.category('routing_mixing', 'Routing & Mixing')}"
CATEGORY_UTILITIES = f"{ROOT_CATEGORY}/{loc.category('utilities', 'Utilities')}"
CATEGORY_SPECTRAL = f"{ROOT_CATEGORY}/{loc.category('spectral_processing', 'Spectral Processing')}"
CATEGORY_MOD_SOURCES = f"{ROOT_CATEGORY}/{loc.category('modulation_sources', 'Modulation Sources')}"
CATEGORY_RESTORATION = f"{ROOT_CATEGORY}/{loc.category('audio_restoration', 'Audio Restoration')}"
CATEGORY_WORKFLOW = f"{ROOT_CATEGORY}/{loc.category('workflow_integration', 'Workflow Integration')}"


def _ui(section: str, name: str, fallback: str | None = None) -> dict:
    return loc.ui(section, name, fallback)


def _audio_input(section: str = "common", name: str = "audio") -> tuple:
    return ("AUDIO", _ui(section, name, "audio"))


def _optional_audio(section: str = "common", name: str = "audio") -> tuple:
    return ("AUDIO", _ui(section, name, name))


def _float(section: str, name: str, default: float, minimum: float, maximum: float, step: float = 0.1) -> tuple:
    cfg = {"default": default, "min": minimum, "max": maximum, "step": step}
    cfg.update(_ui(section, name, name))
    return ("FLOAT", cfg)


def _int(section: str, name: str, default: int, minimum: int, maximum: int, step: int = 1) -> tuple:
    cfg = {"default": default, "min": minimum, "max": maximum, "step": step}
    cfg.update(_ui(section, name, name))
    return ("INT", cfg)


def _bool(section: str, name: str, default: bool) -> tuple:
    cfg = {"default": default}
    cfg.update(_ui(section, name, name))
    return ("BOOLEAN", cfg)


def _string(section: str, name: str, default: str = "", multiline: bool = False) -> tuple:
    cfg = {"default": default, "multiline": multiline}
    cfg.update(_ui(section, name, name))
    return ("STRING", cfg)


def _combo(section: str, name: str, options: list[str], default: str) -> tuple:
    cfg = {"default": default}
    cfg.update(_ui(section, name, name))
    return (options, cfg)


def _force_input(section: str, name: str, type_name: str) -> tuple:
    cfg = {"forceInput": True}
    cfg.update(_ui(section, name, name))
    return (type_name, cfg)


class _AudioDSPNode:
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"

    @classmethod
    def _finish(cls, inputs: dict, section: str, optional: dict | None = None) -> dict:
        config = {"required": {"audio": _audio_input(section), **inputs}}
        if optional:
            config["optional"] = optional
        return config


class ComfyAudioDSPCompressor(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPCompressor", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPCompressor", "Controls dynamic range with threshold, ratio, attack, release, knee, makeup gain, detector, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "compressor"
        return cls._finish({
            "threshold_db": _float(section, "threshold_db", -18.0, -80.0, 0.0),
            "ratio": _float(section, "ratio", 4.0, 1.0, 50.0),
            "attack_ms": _float(section, "attack_ms", 10.0, 0.0, 500.0),
            "release_ms": _float(section, "release_ms", 100.0, 1.0, 5000.0, 1.0),
            "knee_db": _float(section, "knee_db", 6.0, 0.0, 48.0),
            "makeup_gain_db": _float(section, "makeup_gain_db", 0.0, -24.0, 48.0),
            "detector": _combo(section, "detector", ["rms", "peak"], "rms"),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db, detector, mix):
        return (dsp.compressor(audio, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db, detector, mix),)


class ComfyAudioDSPLimiter(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLimiter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPLimiter", "Brickwall-style limiter with threshold, release, and lookahead.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "limiter"
        return cls._finish({
            "threshold_db": _float(section, "threshold_db", -1.0, -36.0, 0.0),
            "release_ms": _float(section, "release_ms", 80.0, 1.0, 2000.0, 1.0),
            "lookahead_ms": _float(section, "lookahead_ms", 2.0, 0.0, 20.0),
        }, section)

    def process(self, audio, threshold_db, release_ms, lookahead_ms):
        return (dsp.limiter(audio, threshold_db, release_ms, lookahead_ms),)


class ComfyAudioDSPMidSideCompressor(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMidSideCompressor", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMidSideCompressor", "Compresses mid and side components independently.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "mid_side_compressor"
        return cls._finish({"mid_threshold_db": _float(section, "mid_threshold_db", -18.0, -80.0, 0.0), "mid_ratio": _float(section, "mid_ratio", 3.0, 1.0, 50.0), "side_threshold_db": _float(section, "side_threshold_db", -22.0, -80.0, 0.0), "side_ratio": _float(section, "side_ratio", 2.0, 1.0, 50.0), "attack_ms": _float(section, "attack_ms", 15.0, 0.0, 500.0), "release_ms": _float(section, "release_ms", 150.0, 1.0, 5000.0, 1.0), "knee_db": _float(section, "knee_db", 6.0, 0.0, 48.0), "side_makeup_db": _float(section, "side_makeup_db", 0.0, -24.0, 24.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, mid_threshold_db, mid_ratio, side_threshold_db, side_ratio, attack_ms, release_ms, knee_db, side_makeup_db, mix):
        return (dsp.mid_side_compressor(audio, mid_threshold_db, mid_ratio, side_threshold_db, side_ratio, attack_ms, release_ms, knee_db, side_makeup_db, mix),)


class ComfyAudioDSPNoiseGate(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPNoiseGate", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPNoiseGate", "Closes the gate below a threshold with attack, hold, release, and attenuation range.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "noise_gate"
        return cls._finish({
            "threshold_db": _float(section, "threshold_db", -45.0, -100.0, 0.0),
            "attack_ms": _float(section, "attack_ms", 5.0, 0.0, 500.0),
            "hold_ms": _float(section, "hold_ms", 50.0, 0.0, 2000.0, 1.0),
            "release_ms": _float(section, "release_ms", 120.0, 1.0, 5000.0, 1.0),
            "range_db": _float(section, "range_db", 60.0, 0.0, 120.0, 0.5),
        }, section)

    def process(self, audio, threshold_db, attack_ms, hold_ms, release_ms, range_db):
        return (dsp.noise_gate(audio, threshold_db, attack_ms, hold_ms, release_ms, range_db),)


class ComfyAudioDSPExpander(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPExpander", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPExpander", "Reduces low-level signals to make background noise and tails quieter.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "expander"
        return cls._finish({
            "threshold_db": _float(section, "threshold_db", -40.0, -100.0, 0.0),
            "ratio": _float(section, "ratio", 2.0, 1.0, 20.0),
            "attack_ms": _float(section, "attack_ms", 10.0, 0.0, 500.0),
            "release_ms": _float(section, "release_ms", 150.0, 1.0, 5000.0, 1.0),
            "range_db": _float(section, "range_db", 40.0, 0.0, 120.0, 0.5),
        }, section)

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, range_db):
        return (dsp.expander(audio, threshold_db, ratio, attack_ms, release_ms, range_db),)


class ComfyAudioDSPTransientShaper(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTransientShaper", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTransientShaper", "Controls attack and sustain gain independently.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "transient_shaper"
        return cls._finish({
            "attack_gain_db": _float(section, "attack_gain_db", 3.0, -24.0, 24.0),
            "sustain_gain_db": _float(section, "sustain_gain_db", 0.0, -24.0, 24.0),
            "fast_ms": _float(section, "fast_ms", 5.0, 0.5, 50.0),
            "slow_ms": _float(section, "slow_ms", 80.0, 10.0, 500.0, 1.0),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, attack_gain_db, sustain_gain_db, fast_ms, slow_ms, mix):
        return (dsp.transient_shaper(audio, attack_gain_db, sustain_gain_db, fast_ms, slow_ms, mix),)


class ComfyAudioDSPDeEsser(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDeEsser", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDeEsser", "Compresses a selected high-frequency band to reduce sibilance.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "de_esser"
        return cls._finish({
            "threshold_db": _float(section, "threshold_db", -30.0, -80.0, 0.0),
            "ratio": _float(section, "ratio", 6.0, 1.0, 30.0),
            "attack_ms": _float(section, "attack_ms", 2.0, 0.0, 100.0),
            "release_ms": _float(section, "release_ms", 80.0, 1.0, 2000.0, 1.0),
            "frequency_low_hz": _float(section, "frequency_low_hz", 4000.0, 500.0, 20000.0, 10.0),
            "frequency_high_hz": _float(section, "frequency_high_hz", 10000.0, 1000.0, 22000.0, 10.0),
            "amount": _float(section, "amount", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, frequency_low_hz, frequency_high_hz, amount):
        return (dsp.de_esser(audio, threshold_db, ratio, attack_ms, release_ms, frequency_low_hz, frequency_high_hz, amount),)


class ComfyAudioDSPMultiBandCompressor(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMultiBandCompressor", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMultiBandCompressor", "Splits audio into 3 or 4 bands and compresses each band independently.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "multiband_compressor"
        return cls._finish({
            "bands": _combo(section, "bands", ["3", "4"], "4"),
            "crossover_low_hz": _float(section, "crossover_low_hz", 160.0, 40.0, 2000.0, 1.0),
            "crossover_mid_hz": _float(section, "crossover_mid_hz", 1200.0, 100.0, 8000.0, 1.0),
            "crossover_high_hz": _float(section, "crossover_high_hz", 6000.0, 1000.0, 20000.0, 1.0),
            "low_threshold_db": _float(section, "low_threshold_db", -20.0, -80.0, 0.0),
            "low_ratio": _float(section, "low_ratio", 3.0, 1.0, 30.0),
            "low_makeup_db": _float(section, "low_makeup_db", 0.0, -24.0, 24.0),
            "low_mid_threshold_db": _float(section, "low_mid_threshold_db", -18.0, -80.0, 0.0),
            "low_mid_ratio": _float(section, "low_mid_ratio", 3.0, 1.0, 30.0),
            "low_mid_makeup_db": _float(section, "low_mid_makeup_db", 0.0, -24.0, 24.0),
            "high_mid_threshold_db": _float(section, "high_mid_threshold_db", -16.0, -80.0, 0.0),
            "high_mid_ratio": _float(section, "high_mid_ratio", 2.5, 1.0, 30.0),
            "high_mid_makeup_db": _float(section, "high_mid_makeup_db", 0.0, -24.0, 24.0),
            "high_threshold_db": _float(section, "high_threshold_db", -18.0, -80.0, 0.0),
            "high_ratio": _float(section, "high_ratio", 2.0, 1.0, 30.0),
            "high_makeup_db": _float(section, "high_makeup_db", 0.0, -24.0, 24.0),
            "attack_ms": _float(section, "attack_ms", 15.0, 0.0, 500.0),
            "release_ms": _float(section, "release_ms", 150.0, 1.0, 5000.0, 1.0),
            "knee_db": _float(section, "knee_db", 6.0, 0.0, 48.0),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz, low_threshold_db, low_ratio, low_makeup_db, low_mid_threshold_db, low_mid_ratio, low_mid_makeup_db, high_mid_threshold_db, high_mid_ratio, high_mid_makeup_db, high_threshold_db, high_ratio, high_makeup_db, attack_ms, release_ms, knee_db, mix):
        return (dsp.multiband_compressor(audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz, low_threshold_db, low_ratio, low_makeup_db, low_mid_threshold_db, low_mid_ratio, low_mid_makeup_db, high_mid_threshold_db, high_mid_ratio, high_mid_makeup_db, high_threshold_db, high_ratio, high_makeup_db, attack_ms, release_ms, knee_db, mix),)


class ComfyAudioDSPMultiBandLimiter(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMultiBandLimiter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMultiBandLimiter", "Splits audio into bands and applies a limiter to each band before summing.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "multiband_limiter"
        return cls._finish({"bands": _combo(section, "bands", ["3", "4"], "4"), "crossover_low_hz": _float(section, "crossover_low_hz", 160.0, 20.0, 20000.0, 1.0), "crossover_mid_hz": _float(section, "crossover_mid_hz", 1200.0, 20.0, 22000.0, 1.0), "crossover_high_hz": _float(section, "crossover_high_hz", 6000.0, 20.0, 22000.0, 1.0), "threshold_db": _float(section, "threshold_db", -1.0, -36.0, 0.0, 0.1), "release_ms": _float(section, "release_ms", 80.0, 1.0, 2000.0, 1.0), "lookahead_ms": _float(section, "lookahead_ms", 2.0, 0.0, 20.0), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz, threshold_db, release_ms, lookahead_ms, mix):
        return (dsp.multiband_limiter(audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz, threshold_db, release_ms, lookahead_ms, mix),)


class ComfyAudioDSPAutoGainLeveler(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAutoGainLeveler", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAutoGainLeveler", "Automatically matches a target RMS or peak level with smoothed gain.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "auto_gain_leveler"
        return cls._finish({
            "mode": _combo(section, "mode", ["RMS", "Peak"], "RMS"),
            "target_db": _float(section, "target_db", -18.0, -60.0, 0.0),
            "window_ms": _float(section, "window_ms", 500.0, 10.0, 10000.0, 10.0),
            "attack_ms": _float(section, "attack_ms", 100.0, 0.0, 5000.0, 1.0),
            "release_ms": _float(section, "release_ms", 1000.0, 1.0, 10000.0, 1.0),
            "min_gain_db": _float(section, "min_gain_db", -24.0, -60.0, 0.0),
            "max_gain_db": _float(section, "max_gain_db", 24.0, 0.0, 60.0),
        }, section)

    def process(self, audio, mode, target_db, window_ms, attack_ms, release_ms, min_gain_db, max_gain_db):
        return (dsp.auto_gain_leveler(audio, mode, target_db, window_ms, attack_ms, release_ms, min_gain_db, max_gain_db),)


class ComfyAudioDSPLoudnessNormalizer(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLoudnessNormalizer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPLoudnessNormalizer", "Normalizes approximate EBU R128 integrated, short-term, or momentary LUFS.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "loudness_normalizer"
        return cls._finish({
            "target_lufs": _float(section, "target_lufs", -16.0, -40.0, 0.0),
            "measurement": _combo(section, "measurement", ["integrated", "short_term", "momentary"], "integrated"),
            "min_gain_db": _float(section, "min_gain_db", -24.0, -60.0, 0.0),
            "max_gain_db": _float(section, "max_gain_db", 24.0, 0.0, 60.0),
            "true_peak_ceiling_db": _float(section, "true_peak_ceiling_db", -1.0, -12.0, 0.0),
        }, section)

    def process(self, audio, target_lufs, measurement, min_gain_db, max_gain_db, true_peak_ceiling_db):
        return (dsp.loudness_normalizer(audio, target_lufs, measurement, min_gain_db, max_gain_db, true_peak_ceiling_db),)


class ComfyAudioDSPLowHighShelf(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLowHighShelf", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPLowHighShelf", "Low or high shelf equalizer with frequency, gain, Q/slope, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "low_high_shelf"
        return cls._finish({
            "shelf": _combo(section, "shelf", ["low_shelf", "high_shelf"], "low_shelf"),
            "frequency_hz": _float(section, "frequency_hz", 200.0, 20.0, 22000.0, 1.0),
            "gain_db": _float(section, "gain_db", 0.0, -24.0, 24.0),
            "q": _float(section, "q", 0.707, 0.1, 10.0, 0.001),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, shelf, frequency_hz, gain_db, q, mix):
        return (dsp.shelf_filter(audio, shelf, frequency_hz, gain_db, q, mix),)


class ComfyAudioDSPPeakBellFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPeakBellFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPeakBellFilter", "Peak/bell equalizer with frequency, gain, Q, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "peak_bell_filter"
        return cls._finish({
            "frequency_hz": _float(section, "frequency_hz", 1000.0, 20.0, 22000.0, 1.0),
            "gain_db": _float(section, "gain_db", 0.0, -24.0, 24.0),
            "q": _float(section, "q", 1.0, 0.1, 30.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, frequency_hz, gain_db, q, mix):
        return (dsp.peak_filter(audio, frequency_hz, gain_db, q, mix),)


class ComfyAudioDSPLowHighPassFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLowHighPassFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPLowHighPassFilter", "Low-pass or high-pass filter with cutoff frequency, order/Q, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "low_high_pass_filter"
        return cls._finish({
            "mode": _combo(section, "mode", ["low_pass", "high_pass"], "low_pass"),
            "cutoff_hz": _float(section, "cutoff_hz", 12000.0, 20.0, 22000.0, 1.0),
            "order": _int(section, "order", 4, 1, 12),
            "q": _float(section, "q", 0.707, 0.1, 20.0, 0.001),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, mode, cutoff_hz, order, q, mix):
        return (dsp.pass_filter(audio, mode, cutoff_hz, order, q, mix),)


class ComfyAudioDSPBandPassStopFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBandPassStopFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPBandPassStopFilter", "Band-pass or band-stop filter with center frequency, bandwidth, order, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "band_pass_stop_filter"
        return cls._finish({
            "mode": _combo(section, "mode", ["band_pass", "band_stop"], "band_pass"),
            "center_hz": _float(section, "center_hz", 1000.0, 20.0, 22000.0, 1.0),
            "bandwidth_hz": _float(section, "bandwidth_hz", 1000.0, 20.0, 22000.0, 1.0),
            "order": _int(section, "order", 4, 1, 12),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, mode, center_hz, bandwidth_hz, order, mix):
        return (dsp.band_filter(audio, mode, center_hz, bandwidth_hz, order, mix),)


class ComfyAudioDSPNotchFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPNotchFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPNotchFilter", "Very narrow band-stop filter for removing a single frequency.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "notch_filter"
        return cls._finish({
            "frequency_hz": _float(section, "frequency_hz", 50.0, 20.0, 22000.0, 1.0),
            "q": _float(section, "q", 30.0, 1.0, 500.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, frequency_hz, q, mix):
        return (dsp.notch_filter(audio, frequency_hz, q, mix),)


class ComfyAudioDSPThreeBandEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPThreeBandEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPThreeBandEQ", "Classic 3-band EQ with low shelf, mid peak, and high shelf.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "three_band_eq"
        return cls._finish({
            "low_frequency_hz": _float(section, "low_frequency_hz", 120.0, 20.0, 2000.0, 1.0),
            "low_gain_db": _float(section, "low_gain_db", 0.0, -24.0, 24.0),
            "mid_frequency_hz": _float(section, "mid_frequency_hz", 1000.0, 100.0, 10000.0, 1.0),
            "mid_gain_db": _float(section, "mid_gain_db", 0.0, -24.0, 24.0),
            "mid_q": _float(section, "mid_q", 1.0, 0.1, 30.0, 0.01),
            "high_frequency_hz": _float(section, "high_frequency_hz", 8000.0, 1000.0, 22000.0, 1.0),
            "high_gain_db": _float(section, "high_gain_db", 0.0, -24.0, 24.0),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, low_frequency_hz, low_gain_db, mid_frequency_hz, mid_gain_db, mid_q, high_frequency_hz, high_gain_db, mix):
        return (dsp.three_band_eq(audio, low_frequency_hz, low_gain_db, mid_frequency_hz, mid_gain_db, mid_q, high_frequency_hz, high_gain_db, mix),)


class ComfyAudioDSPParametricEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPParametricEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPParametricEQ", "Configurable parametric EQ with up to 8 bands.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "parametric_eq"
        required = {
            "audio": _audio_input(section),
            "bands": _int(section, "bands", 4, 1, 8),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }
        defaults = [(80.0, 0.0, 0.707), (250.0, 0.0, 1.0), (1000.0, 0.0, 1.0), (4000.0, 0.0, 1.0), (10000.0, 0.0, 0.707), (16000.0, 0.0, 0.707), (500.0, 0.0, 1.0), (2000.0, 0.0, 1.0)]
        for index, (freq, gain, q) in enumerate(defaults, start=1):
            required[f"band_{index}_type"] = _combo(section, f"band_{index}_type", FILTER_TYPES, "peak")
            required[f"band_{index}_frequency_hz"] = _float(section, f"band_{index}_frequency_hz", freq, 20.0, 22000.0, 1.0)
            required[f"band_{index}_gain_db"] = _float(section, f"band_{index}_gain_db", gain, -24.0, 24.0)
            required[f"band_{index}_q"] = _float(section, f"band_{index}_q", q, 0.1, 30.0, 0.01)
        return {"required": required}

    def process(self, audio, bands, mix, **kwargs):
        values = []
        for index in range(1, 9):
            values.extend([
                kwargs[f"band_{index}_type"],
                kwargs[f"band_{index}_frequency_hz"],
                kwargs[f"band_{index}_gain_db"],
                kwargs[f"band_{index}_q"],
            ])
        return (dsp.parametric_eq(audio, bands, mix, *values),)


class ComfyAudioDSPGraphicEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPGraphicEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPGraphicEQ", "Graphic EQ with 10, 15, or 31 fixed-frequency gain bands.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "graphic_eq"
        required = {
            "audio": _audio_input(section),
            "bands": _combo(section, "bands", ["10", "15", "31"], "10"),
            "q": _float(section, "q", 1.414, 0.3, 10.0, 0.001),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }
        for index, center in enumerate(GRAPHIC_EQ_BANDS["31"], start=1):
            required[f"gain_{index:02d}_db"] = _float(section, f"gain_{index:02d}_db", 0.0, -18.0, 18.0)
        return {"required": required}

    def process(self, audio, bands, q, mix, **kwargs):
        gains = [kwargs[f"gain_{index:02d}_db"] for index in range(1, 32)]
        return (dsp.graphic_eq(audio, bands, q, mix, *gains),)


class ComfyAudioDSPTiltEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTiltEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTiltEQ", "Tilt EQ around a pivot frequency, raising highs while lowering lows or the reverse.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "tilt_eq"
        return cls._finish({
            "pivot_hz": _float(section, "pivot_hz", 1000.0, 50.0, 12000.0, 1.0),
            "tilt_db": _float(section, "tilt_db", 0.0, -24.0, 24.0),
            "q": _float(section, "q", 0.707, 0.1, 10.0, 0.001),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, pivot_hz, tilt_db, q, mix):
        return (dsp.tilt_eq(audio, pivot_hz, tilt_db, q, mix),)


class ComfyAudioDSPRiaaEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPRiaaEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPRiaaEQ", "RIAA phono equalization with de-emphasis/playback and pre-emphasis modes.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "riaa_eq"
        return cls._finish({
            "mode": _combo(section, "mode", ["de_emphasis", "pre_emphasis"], "de_emphasis"),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, mode, mix):
        return (dsp.riaa_eq(audio, mode, mix),)


class ComfyAudioDSPResonantPassFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPResonantPassFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPResonantPassFilter", "Low-pass or high-pass filter with resonance for synth-style filtering.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "resonant_pass_filter"
        return cls._finish({
            "mode": _combo(section, "mode", ["low_pass", "high_pass"], "low_pass"),
            "cutoff_hz": _float(section, "cutoff_hz", 1000.0, 20.0, 22000.0, 1.0),
            "resonance_q": _float(section, "resonance_q", 1.0, 0.1, 30.0, 0.01),
            "drive_db": _float(section, "drive_db", 0.0, 0.0, 24.0),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, mode, cutoff_hz, resonance_q, drive_db, mix):
        return (dsp.resonant_pass_filter(audio, mode, cutoff_hz, resonance_q, drive_db, mix),)


class ComfyAudioDSPMatchEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMatchEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMatchEQ", "Matches the source spectral envelope toward a reference audio signal.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "match_eq"
        return {"required": {"audio": _audio_input(section), "reference_audio": _audio_input(section, "reference_audio"), "amount": _float(section, "amount", 0.75, 0.0, 1.0, 0.01), "smoothing_bins": _int(section, "smoothing_bins", 31, 1, 2048, 2), "max_gain_db": _float(section, "max_gain_db", 12.0, 0.0, 48.0, 0.1), "fft_size": _int(section, "fft_size", 8192, 256, 262144, 256), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}}

    def process(self, audio, reference_audio, amount, smoothing_bins, max_gain_db, fft_size, mix):
        return (dsp.match_eq(audio, reference_audio, amount, smoothing_bins, max_gain_db, fft_size, mix),)


class ComfyAudioDSPConvolutionReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPConvolutionReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPConvolutionReverb", "Convolution reverb from a WAV impulse response with dry/wet controls.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "convolution_reverb"
        return cls._finish({
            "impulse_response_wav": _string(section, "impulse_response_wav"),
            "pre_delay_ms": _float(section, "pre_delay_ms", 0.0, 0.0, 500.0),
            "wet": _float(section, "wet", 0.35, 0.0, 2.0, 0.01),
            "dry": _float(section, "dry", 1.0, 0.0, 2.0, 0.01),
            "normalize_ir": _bool(section, "normalize_ir", True),
        }, section)

    def process(self, audio, impulse_response_wav, pre_delay_ms, wet, dry, normalize_ir):
        return (dsp.convolution_reverb(audio, impulse_response_wav, pre_delay_ms, wet, dry, normalize_ir),)


class ComfyAudioDSPIRManager:
    CATEGORY = CATEGORY_REVERB
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPIRManager", ("ir_audio", "info"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPIRManager", "Loads, trims, reverses, normalizes, and reports impulse-response WAV files.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "ir_manager"
        return {"required": {"path": _string(section, "path"), "target_sample_rate": _int(section, "target_sample_rate", 44100, 1000, 384000, 1), "start_ms": _float(section, "start_ms", 0.0, 0.0, 60000.0, 0.1), "max_duration_s": _float(section, "max_duration_s", 0.0, 0.0, 600.0, 0.001), "normalize_ir": _bool(section, "normalize_ir", True), "reverse": _bool(section, "reverse", False)}}

    def process(self, path, target_sample_rate, start_ms, max_duration_s, normalize_ir, reverse):
        return dsp.ir_manager(path, target_sample_rate, start_ms, max_duration_s, normalize_ir, reverse)


class ComfyAudioDSPSchroederReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSchroederReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSchroederReverb", "Classic Schroeder reverb with pre-delay, decay, diffusion, and tone filters.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "schroeder_reverb"
        return cls._finish({
            "pre_delay_ms": _float(section, "pre_delay_ms", 20.0, 0.0, 500.0),
            "decay_time_s": _float(section, "decay_time_s", 1.6, 0.1, 20.0),
            "diffusion": _float(section, "diffusion", 0.65, 0.0, 1.0, 0.01),
            "low_cut_hz": _float(section, "low_cut_hz", 120.0, 20.0, 2000.0, 1.0),
            "high_cut_hz": _float(section, "high_cut_hz", 12000.0, 1000.0, 22000.0, 1.0),
            "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, pre_delay_ms, decay_time_s, diffusion, low_cut_hz, high_cut_hz, mix):
        return (dsp.schroeder_reverb(audio, pre_delay_ms, decay_time_s, diffusion, low_cut_hz, high_cut_hz, mix),)


class ComfyAudioDSPFreeverbMoorerReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFreeverbMoorerReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFreeverbMoorerReverb", "Feedback-delay-network style reverb with damping, stereo width, and early reflections.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "freeverb_moorer_reverb"
        return cls._finish({
            "room_size": _float(section, "room_size", 0.55, 0.0, 1.0, 0.01),
            "damping": _float(section, "damping", 0.35, 0.0, 1.0, 0.01),
            "width": _float(section, "width", 1.0, 0.0, 2.0, 0.01),
            "early_reflections": _float(section, "early_reflections", 0.35, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, room_size, damping, width, early_reflections, mix):
        return (dsp.freeverb_moorer_reverb(audio, room_size, damping, width, early_reflections, mix),)


class ComfyAudioDSPSpringReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpringReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSpringReverb", "Spring reverb simulation with all-pass and delay-network dispersion.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spring_reverb"
        return cls._finish({
            "tension": _float(section, "tension", 0.5, 0.0, 1.0, 0.01),
            "decay_time_s": _float(section, "decay_time_s", 1.8, 0.1, 10.0),
            "tone_hz": _float(section, "tone_hz", 1800.0, 100.0, 12000.0, 1.0),
            "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, tension, decay_time_s, tone_hz, mix):
        return (dsp.spring_reverb(audio, tension, decay_time_s, tone_hz, mix),)


class ComfyAudioDSPPlateReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPlateReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPlateReverb", "Plate reverb simulation with dense diffusion and smooth decay.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "plate_reverb"
        return cls._finish({
            "decay_time_s": _float(section, "decay_time_s", 2.2, 0.1, 20.0),
            "diffusion": _float(section, "diffusion", 0.8, 0.0, 1.0, 0.01),
            "damping": _float(section, "damping", 0.25, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, decay_time_s, diffusion, damping, mix):
        return (dsp.plate_reverb(audio, decay_time_s, diffusion, damping, mix),)


class ComfyAudioDSPGatedReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPGatedReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPGatedReverb", "Gated reverb with an abrupt nonlinear tail cutoff for 1980s-style ambience.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "gated_reverb"
        return cls._finish({
            "reverb_time_s": _float(section, "reverb_time_s", 1.8, 0.1, 10.0),
            "gate_time_ms": _float(section, "gate_time_ms", 450.0, 20.0, 3000.0, 1.0),
            "release_ms": _float(section, "release_ms", 40.0, 1.0, 500.0, 1.0),
            "mix": _float(section, "mix", 0.3, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, reverb_time_s, gate_time_ms, release_ms, mix):
        return (dsp.gated_reverb(audio, reverb_time_s, gate_time_ms, release_ms, mix),)


class ComfyAudioDSPReverseReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPReverseReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPReverseReverb", "Reverse audio, apply reverb, then reverse back to create a swelling lead-in effect.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "reverse_reverb"
        return cls._finish({
            "reverb_time_s": _float(section, "reverb_time_s", 1.8, 0.1, 10.0),
            "diffusion": _float(section, "diffusion", 0.75, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, reverb_time_s, diffusion, mix):
        return (dsp.reverse_reverb(audio, reverb_time_s, diffusion, mix),)


class ComfyAudioDSPSimpleDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSimpleDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSimpleDelay", "Basic delay with delay time, feedback, and dry/wet mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "simple_delay"
        return cls._finish({
            "delay_ms": _float(section, "delay_ms", 350.0, 1.0, 5000.0, 1.0),
            "feedback": _float(section, "feedback", 0.35, -0.95, 0.95, 0.01),
            "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, delay_ms, feedback, mix):
        return (dsp.simple_delay(audio, delay_ms, feedback, mix),)


class ComfyAudioDSPTempoSyncedDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTempoSyncedDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTempoSyncedDelay", "Tempo-synced delay using BPM and musical note values.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "tempo_synced_delay"
        return cls._finish({
            "bpm": _float(section, "bpm", 120.0, 20.0, 300.0, 0.1),
            "note_value": _combo(section, "note_value", list(NOTE_VALUES.keys()), "1/4"),
            "feedback": _float(section, "feedback", 0.35, -0.95, 0.95, 0.01),
            "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01),
        }, section, {
            "song_bpm": _force_input(section, "song_bpm", "FLOAT"),
            "analysis_json": _force_input(section, "analysis_json", "STRING"),
        })

    def process(self, audio, bpm, note_value, feedback, mix, song_bpm=None, analysis_json=None):
        bpm = dsp.song_bpm_value(bpm, song_bpm, analysis_json)
        return (dsp.tempo_synced_delay(audio, bpm, note_value, feedback, mix),)


class ComfyAudioDSPPingPongDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPingPongDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPingPongDelay", "Ping-pong delay with alternating left/right feedback.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "ping_pong_delay"
        return cls._finish({
            "delay_ms": _float(section, "delay_ms", 320.0, 1.0, 5000.0, 1.0),
            "feedback": _float(section, "feedback", 0.4, -0.95, 0.95, 0.01),
            "width": _float(section, "width", 1.0, 0.0, 2.0, 0.01),
            "mix": _float(section, "mix", 0.3, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, delay_ms, feedback, width, mix):
        return (dsp.ping_pong_delay(audio, delay_ms, feedback, width, mix),)


class ComfyAudioDSPMultiTapDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMultiTapDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMultiTapDelay", "Multi-tap delay with independent delay time and gain per tap.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "multi_tap_delay"
        inputs = {"mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01)}
        defaults = [(90.0, 0.45), (180.0, 0.35), (270.0, 0.28), (420.0, 0.22), (620.0, 0.16), (850.0, 0.10)]
        for index, (time_ms, gain) in enumerate(defaults, start=1):
            inputs[f"tap_{index}_ms"] = _float(section, f"tap_{index}_ms", time_ms, 0.0, 5000.0, 1.0)
            inputs[f"tap_{index}_gain"] = _float(section, f"tap_{index}_gain", gain, -2.0, 2.0, 0.01)
        return cls._finish(inputs, section)

    def process(self, audio, mix, **kwargs):
        values = []
        for index in range(1, 7):
            values.extend([kwargs[f"tap_{index}_ms"], kwargs[f"tap_{index}_gain"]])
        return (dsp.multi_tap_delay(audio, mix, *values),)


class ComfyAudioDSPDubDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDubDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDubDelay", "Tape/BBD-style dub delay with low-passed feedback and wow/flutter.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "dub_delay"
        return cls._finish({
            "delay_ms": _float(section, "delay_ms", 420.0, 1.0, 5000.0, 1.0),
            "feedback": _float(section, "feedback", 0.55, -0.95, 0.95, 0.01),
            "tone_hz": _float(section, "tone_hz", 4200.0, 200.0, 16000.0, 1.0),
            "wow_depth_ms": _float(section, "wow_depth_ms", 2.0, 0.0, 20.0, 0.1),
            "wow_rate_hz": _float(section, "wow_rate_hz", 0.45, 0.01, 10.0, 0.01),
            "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, delay_ms, feedback, tone_hz, wow_depth_ms, wow_rate_hz, mix):
        return (dsp.dub_delay(audio, delay_ms, feedback, tone_hz, wow_depth_ms, wow_rate_hz, mix),)


class ComfyAudioDSPFilteredDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFilteredDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFilteredDelay", "Delay with high-pass and low-pass filtering on the delayed signal.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "filtered_delay"
        return cls._finish({
            "delay_ms": _float(section, "delay_ms", 360.0, 1.0, 5000.0, 1.0),
            "feedback": _float(section, "feedback", 0.4, -0.95, 0.95, 0.01),
            "low_cut_hz": _float(section, "low_cut_hz", 180.0, 20.0, 4000.0, 1.0),
            "high_cut_hz": _float(section, "high_cut_hz", 7000.0, 500.0, 22000.0, 1.0),
            "mix": _float(section, "mix", 0.3, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, delay_ms, feedback, low_cut_hz, high_cut_hz, mix):
        return (dsp.filtered_delay(audio, delay_ms, feedback, low_cut_hz, high_cut_hz, mix),)


class ComfyAudioDSPStereoSpreadDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPStereoSpreadDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPStereoSpreadDelay", "Stereo spread delay using small left/right delay offsets for width.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "stereo_spread_delay"
        return cls._finish({
            "base_delay_ms": _float(section, "base_delay_ms", 24.0, 1.0, 500.0, 0.1),
            "spread_ms": _float(section, "spread_ms", 12.0, 0.0, 200.0, 0.1),
            "feedback": _float(section, "feedback", 0.05, -0.95, 0.95, 0.01),
            "width": _float(section, "width", 1.5, 0.0, 2.0, 0.01),
            "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, base_delay_ms, spread_ms, feedback, width, mix):
        return (dsp.stereo_spread_delay(audio, base_delay_ms, spread_ms, feedback, width, mix),)


class ComfyAudioDSPChorus(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPChorus", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPChorus", "Chorus with multiple slightly delayed voices, depth, rate, and feedback.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "chorus"
        return cls._finish({
            "voices": _int(section, "voices", 3, 1, 8),
            "delay_ms": _float(section, "delay_ms", 18.0, 1.0, 80.0, 0.1),
            "depth_ms": _float(section, "depth_ms", 6.0, 0.0, 40.0, 0.1),
            "rate_hz": _float(section, "rate_hz", 0.35, 0.01, 20.0, 0.01),
            "feedback": _float(section, "feedback", 0.05, -0.95, 0.95, 0.01),
            "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, voices, delay_ms, depth_ms, rate_hz, feedback, mix):
        return (dsp.chorus(audio, voices, delay_ms, depth_ms, rate_hz, feedback, mix),)


class ComfyAudioDSPFlanger(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFlanger", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFlanger", "Flanger with short modulated delay, feedback, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "flanger"
        return cls._finish({
            "base_delay_ms": _float(section, "base_delay_ms", 1.8, 0.1, 20.0, 0.01),
            "depth_ms": _float(section, "depth_ms", 2.5, 0.0, 20.0, 0.01),
            "rate_hz": _float(section, "rate_hz", 0.25, 0.01, 20.0, 0.01),
            "feedback": _float(section, "feedback", 0.45, -0.95, 0.95, 0.01),
            "mix": _float(section, "mix", 0.5, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, base_delay_ms, depth_ms, rate_hz, feedback, mix):
        return (dsp.flanger(audio, base_delay_ms, depth_ms, rate_hz, feedback, mix),)


class ComfyAudioDSPPhaser(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPhaser", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPhaser", "Phaser built from cascaded all-pass filters with LFO modulation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "phaser"
        return cls._finish({
            "stages": _int(section, "stages", 6, 2, 12, 2),
            "rate_hz": _float(section, "rate_hz", 0.35, 0.01, 20.0, 0.01),
            "depth": _float(section, "depth", 0.75, 0.0, 1.0, 0.01),
            "feedback": _float(section, "feedback", 0.25, -0.95, 0.95, 0.01),
            "mix": _float(section, "mix", 0.5, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, stages, rate_hz, depth, feedback, mix):
        return (dsp.phaser(audio, stages, rate_hz, depth, feedback, mix),)


class ComfyAudioDSPTremolo(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTremolo", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTremolo", "Amplitude LFO modulation with selectable waveform.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "tremolo"
        return cls._finish({
            "rate_hz": _float(section, "rate_hz", 5.0, 0.01, 30.0, 0.01),
            "depth": _float(section, "depth", 0.75, 0.0, 1.0, 0.01),
            "waveform_shape": _combo(section, "waveform_shape", WAVEFORMS, "sine"),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, rate_hz, depth, waveform_shape, mix):
        return (dsp.tremolo(audio, rate_hz, depth, waveform_shape, mix),)


class ComfyAudioDSPVibrato(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPVibrato", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPVibrato", "Small pitch modulation using an LFO-driven fractional delay.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "vibrato"
        return cls._finish({
            "depth_ms": _float(section, "depth_ms", 3.0, 0.0, 30.0, 0.01),
            "rate_hz": _float(section, "rate_hz", 5.0, 0.01, 30.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, depth_ms, rate_hz, mix):
        return (dsp.vibrato(audio, depth_ms, rate_hz, mix),)


class ComfyAudioDSPRotarySpeaker(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPRotarySpeaker", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPRotarySpeaker", "Leslie-style rotary speaker with independent low rotor and high horn rates.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "rotary_speaker"
        return cls._finish({
            "low_rate_hz": _float(section, "low_rate_hz", 0.7, 0.01, 20.0, 0.01),
            "high_rate_hz": _float(section, "high_rate_hz", 6.5, 0.01, 20.0, 0.01),
            "depth": _float(section, "depth", 0.65, 0.0, 1.0, 0.01),
            "crossover_hz": _float(section, "crossover_hz", 800.0, 100.0, 5000.0, 1.0),
            "mix": _float(section, "mix", 0.65, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, low_rate_hz, high_rate_hz, depth, crossover_hz, mix):
        return (dsp.rotary_speaker(audio, low_rate_hz, high_rate_hz, depth, crossover_hz, mix),)


class ComfyAudioDSPRingModulator(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPRingModulator", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPRingModulator", "Ring modulation by multiplying the signal with a carrier oscillator.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "ring_modulator"
        return cls._finish({
            "carrier_hz": _float(section, "carrier_hz", 440.0, 0.1, 12000.0, 0.1),
            "depth": _float(section, "depth", 1.0, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 0.6, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, carrier_hz, depth, mix):
        return (dsp.ring_modulator(audio, carrier_hz, depth, mix),)


class ComfyAudioDSPAutoPanner(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAutoPanner", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAutoPanner", "LFO-controlled stereo panning.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "auto_panner"
        return cls._finish({
            "rate_hz": _float(section, "rate_hz", 0.7, 0.01, 30.0, 0.01),
            "depth": _float(section, "depth", 1.0, 0.0, 1.0, 0.01),
            "waveform_shape": _combo(section, "waveform_shape", WAVEFORMS, "sine"),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, rate_hz, depth, waveform_shape, mix):
        return (dsp.auto_panner(audio, rate_hz, depth, waveform_shape, mix),)


class ComfyAudioDSPUniVibe(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPUniVibe", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPUniVibe", "Vintage optical phaser/chorus style effect.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "uni_vibe"
        return cls._finish({
            "rate_hz": _float(section, "rate_hz", 1.2, 0.01, 20.0, 0.01),
            "depth": _float(section, "depth", 0.8, 0.0, 1.0, 0.01),
            "chorus_mix": _float(section, "chorus_mix", 0.25, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 0.7, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, rate_hz, depth, chorus_mix, mix):
        return (dsp.uni_vibe(audio, rate_hz, depth, chorus_mix, mix),)


class ComfyAudioDSPSoftClipper(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSoftClipper", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSoftClipper", "Soft clipping with tanh or cubic transfer curves.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "soft_clipper"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 12.0, 0.0, 60.0, 0.1),
            "curve": _combo(section, "curve", ["tanh", "cubic"], "tanh"),
            "output_gain_db": _float(section, "output_gain_db", -6.0, -60.0, 24.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, curve, output_gain_db, mix):
        return (dsp.soft_clipper(audio, drive_db, curve, output_gain_db, mix),)


class ComfyAudioDSPHardClipper(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHardClipper", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHardClipper", "Hard clipping with a simple threshold.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "hard_clipper"
        return cls._finish({
            "threshold": _float(section, "threshold", 0.6, 0.01, 1.0, 0.01),
            "output_gain_db": _float(section, "output_gain_db", -3.0, -60.0, 24.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, threshold, output_gain_db, mix):
        return (dsp.hard_clipper(audio, threshold, output_gain_db, mix),)


class ComfyAudioDSPTubeSaturation(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTubeSaturation", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTubeSaturation", "Tube-style asymmetric waveshaping for even harmonics.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "tube_saturation"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 10.0, 0.0, 60.0, 0.1),
            "asymmetry": _float(section, "asymmetry", 0.35, 0.0, 0.95, 0.01),
            "bias": _float(section, "bias", 0.05, -1.0, 1.0, 0.01),
            "output_gain_db": _float(section, "output_gain_db", -4.0, -60.0, 24.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, asymmetry, bias, output_gain_db, mix):
        return (dsp.tube_saturation(audio, drive_db, asymmetry, bias, output_gain_db, mix),)


class ComfyAudioDSPTapeSaturation(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTapeSaturation", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTapeSaturation", "Tape-style saturation with compression, tone shaping, and light wow.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "tape_saturation"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 8.0, 0.0, 60.0, 0.1),
            "compression": _float(section, "compression", 0.35, 0.0, 1.0, 0.01),
            "tone": _float(section, "tone", 0.65, 0.0, 1.0, 0.01),
            "wow": _float(section, "wow", 0.15, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, compression, tone, wow, mix):
        return (dsp.tape_saturation(audio, drive_db, compression, tone, wow, mix),)


class ComfyAudioDSPFuzz(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFuzz", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFuzz", "Aggressive fuzz distortion with square-like clipping.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "fuzz"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 24.0, 0.0, 80.0, 0.1),
            "gate": _float(section, "gate", 0.0, 0.0, 0.2, 0.001),
            "tone": _float(section, "tone", 0.5, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, gate, tone, mix):
        return (dsp.fuzz(audio, drive_db, gate, tone, mix),)


class ComfyAudioDSPBitCrusher(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBitCrusher", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPBitCrusher", "Bit crusher that lowers quantization depth and sample-rate resolution.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "bit_crusher"
        return cls._finish({
            "bit_depth": _int(section, "bit_depth", 8, 1, 16),
            "downsample_factor": _int(section, "downsample_factor", 4, 1, 128),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, bit_depth, downsample_factor, mix):
        return (dsp.bit_crusher(audio, bit_depth, downsample_factor, mix),)


class ComfyAudioDSPOverdriveDistortion(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPOverdriveDistortion", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPOverdriveDistortion", "Overdrive or distortion with gain, tone, output gain, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "overdrive_distortion"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 14.0, 0.0, 80.0, 0.1),
            "tone": _float(section, "tone", 0.6, 0.0, 1.0, 0.01),
            "mode": _combo(section, "mode", ["overdrive", "distortion"], "overdrive"),
            "output_gain_db": _float(section, "output_gain_db", -6.0, -60.0, 24.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, tone, mode, output_gain_db, mix):
        return (dsp.overdrive_distortion(audio, drive_db, tone, mode, output_gain_db, mix),)


class ComfyAudioDSPWavefolder(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPWavefolder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPWavefolder", "Wavefolding distortion that folds the waveform into complex harmonics.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "wavefolder"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 12.0, 0.0, 60.0, 0.1),
            "folds": _float(section, "folds", 2.0, 1.0, 12.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, folds, mix):
        return (dsp.wavefolder(audio, drive_db, folds, mix),)


class ComfyAudioDSPExciterEnhancer(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPExciterEnhancer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPExciterEnhancer", "Exciter/enhancer that generates harmonics to add high-frequency brightness.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "exciter_enhancer"
        return cls._finish({
            "drive_db": _float(section, "drive_db", 12.0, 0.0, 60.0, 0.1),
            "crossover_hz": _float(section, "crossover_hz", 3500.0, 500.0, 16000.0, 1.0),
            "amount": _float(section, "amount", 0.35, 0.0, 2.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, drive_db, crossover_hz, amount, mix):
        return (dsp.exciter_enhancer(audio, drive_db, crossover_hz, amount, mix),)


class ComfyAudioDSPPitchShifter(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPitchShifter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPitchShifter", "Shifts pitch by semitones and cents while keeping the original duration.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "pitch_shifter"
        return cls._finish({
            "semitones": _float(section, "semitones", 0.0, -24.0, 24.0, 0.01),
            "cents": _float(section, "cents", 0.0, -100.0, 100.0, 1.0),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, semitones, cents, mix):
        return (dsp.pitch_shifter(audio, semitones, cents, mix),)


class ComfyAudioDSPTimeStretcher(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTimeStretcher", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPTimeStretcher", "Changes duration without intentionally changing pitch.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "time_stretcher"
        return cls._finish({
            "time_ratio": _float(section, "time_ratio", 1.0, 0.5, 2.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, time_ratio, mix):
        return (dsp.time_stretcher(audio, time_ratio, mix),)


class ComfyAudioDSPResamplerClassic(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPResamplerClassic", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPResamplerClassic", "Classic tape-style resampling where speed, pitch, and duration change together.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "resampler_classic"
        return cls._finish({
            "speed_ratio": _float(section, "speed_ratio", 1.0, 0.25, 4.0, 0.01),
            "output_gain_db": _float(section, "output_gain_db", 0.0, -24.0, 24.0, 0.1),
        }, section)

    def process(self, audio, speed_ratio, output_gain_db):
        return (dsp.resampler_classic(audio, speed_ratio, output_gain_db),)


class ComfyAudioDSPHarmonizer(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHarmonizer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHarmonizer", "Creates up to four fixed-interval harmony voices.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "harmonizer"
        return cls._finish({
            "voice_1_semitones": _float(section, "voice_1_semitones", 3.0, -24.0, 24.0, 0.01),
            "voice_1_gain": _float(section, "voice_1_gain", 0.5, 0.0, 2.0, 0.01),
            "voice_2_semitones": _float(section, "voice_2_semitones", 7.0, -24.0, 24.0, 0.01),
            "voice_2_gain": _float(section, "voice_2_gain", 0.35, 0.0, 2.0, 0.01),
            "voice_3_semitones": _float(section, "voice_3_semitones", 0.0, -24.0, 24.0, 0.01),
            "voice_3_gain": _float(section, "voice_3_gain", 0.0, 0.0, 2.0, 0.01),
            "voice_4_semitones": _float(section, "voice_4_semitones", 0.0, -24.0, 24.0, 0.01),
            "voice_4_gain": _float(section, "voice_4_gain", 0.0, 0.0, 2.0, 0.01),
            "dry_gain": _float(section, "dry_gain", 1.0, 0.0, 2.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, voice_1_semitones, voice_1_gain, voice_2_semitones, voice_2_gain, voice_3_semitones, voice_3_gain, voice_4_semitones, voice_4_gain, dry_gain, mix):
        return (dsp.harmonizer(audio, voice_1_semitones, voice_1_gain, voice_2_semitones, voice_2_gain, voice_3_semitones, voice_3_gain, voice_4_semitones, voice_4_gain, dry_gain, mix),)


class ComfyAudioDSPPitchCorrection(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPitchCorrection", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPitchCorrection", "Auto-Tune-style monophonic pitch correction to a selected key and scale.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "pitch_correction"
        return cls._finish({
            "key": _combo(section, "key", PITCH_KEYS, "C"),
            "scale": _combo(section, "scale", PITCH_SCALES, "major"),
            "correction_speed": _float(section, "correction_speed", 0.75, 0.0, 1.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section, {
            "song_key": _force_input(section, "song_key", "STRING"),
            "analysis_json": _force_input(section, "analysis_json", "STRING"),
        })

    def process(self, audio, key, scale, correction_speed, mix, song_key=None, analysis_json=None):
        if song_key not in (None, "") or analysis_json not in (None, ""):
            key, scale, _details = dsp.song_key_to_pitch_controls(song_key or "", analysis_json)
        return (dsp.pitch_correction(audio, key, scale, correction_speed, mix),)


class ComfyAudioDSPVarispeedPlayer(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPVarispeedPlayer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPVarispeedPlayer", "Varispeed playback that changes speed, duration, and pitch together.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "varispeed_player"
        return cls._finish({
            "speed_ratio": _float(section, "speed_ratio", 1.0, 0.25, 4.0, 0.01),
            "output_gain_db": _float(section, "output_gain_db", 0.0, -24.0, 24.0, 0.1),
        }, section)

    def process(self, audio, speed_ratio, output_gain_db):
        return (dsp.varispeed_player(audio, speed_ratio, output_gain_db),)


class ComfyAudioDSPPannerBalance(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPannerBalance", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPannerBalance", "Left/right balance panner with optional equal-power law.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "panner_balance"
        return cls._finish({
            "pan": _float(section, "pan", 0.0, -1.0, 1.0, 0.01),
            "equal_power": _bool(section, "equal_power", True),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, pan, equal_power, mix):
        return (dsp.panner_balance(audio, pan, equal_power, mix),)


class ComfyAudioDSPStereoWidth(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPStereoWidth", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPStereoWidth", "Widens or narrows the stereo image with mid/side processing.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "stereo_width"
        return cls._finish({
            "width": _float(section, "width", 1.0, 0.0, 3.0, 0.01),
            "gain_db": _float(section, "gain_db", 0.0, -24.0, 24.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, width, gain_db, mix):
        return (dsp.stereo_width(audio, width, gain_db, mix),)


class ComfyAudioDSPMidSideEncoder(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMidSideEncoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMidSideEncoder", "Encodes left/right stereo into mid and side channels.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "mid_side_encoder"
        return cls._finish({"normalize": _bool(section, "normalize", False)}, section)

    def process(self, audio, normalize):
        return (dsp.mid_side_encoder(audio, normalize),)


class ComfyAudioDSPMidSideDecoder(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMidSideDecoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMidSideDecoder", "Decodes mid/side channels back to left/right stereo.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "mid_side_decoder"
        return cls._finish({"normalize": _bool(section, "normalize", False)}, section)

    def process(self, audio, normalize):
        return (dsp.mid_side_decoder(audio, normalize),)


class ComfyAudioDSPMidSideEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMidSideEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMidSideEQ", "Applies independent filter bands to mid and side signals.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "mid_side_eq"
        return cls._finish({
            "mid_filter_type": _combo(section, "mid_filter_type", MID_SIDE_EQ_FILTER_TYPES, "peak"),
            "mid_frequency_hz": _float(section, "mid_frequency_hz", 1000.0, 20.0, 22000.0, 1.0),
            "mid_gain_db": _float(section, "mid_gain_db", 0.0, -24.0, 24.0),
            "mid_q": _float(section, "mid_q", 1.0, 0.1, 30.0, 0.01),
            "side_filter_type": _combo(section, "side_filter_type", MID_SIDE_EQ_FILTER_TYPES, "peak"),
            "side_frequency_hz": _float(section, "side_frequency_hz", 4000.0, 20.0, 22000.0, 1.0),
            "side_gain_db": _float(section, "side_gain_db", 0.0, -24.0, 24.0),
            "side_q": _float(section, "side_q", 1.0, 0.1, 30.0, 0.01),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, mid_filter_type, mid_frequency_hz, mid_gain_db, mid_q, side_filter_type, side_frequency_hz, side_gain_db, side_q, mix):
        return (dsp.mid_side_eq(audio, mid_filter_type, mid_frequency_hz, mid_gain_db, mid_q, side_filter_type, side_frequency_hz, side_gain_db, side_q, mix),)


class ComfyAudioDSPStereoEnhancerHaas(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPStereoEnhancerHaas", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPStereoEnhancerHaas", "Stereo enhancer using the Haas effect with a short channel delay.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "stereo_enhancer_haas"
        return cls._finish({
            "delay_ms": _float(section, "delay_ms", 12.0, 0.0, 40.0, 0.1),
            "side": _combo(section, "side", ["right", "left"], "right"),
            "feedback": _float(section, "feedback", 0.5, -1.0, 1.0, 0.01),
            "mix": _float(section, "mix", 0.65, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, delay_ms, side, feedback, mix):
        return (dsp.stereo_enhancer_haas(audio, delay_ms, side, feedback, mix),)


class ComfyAudioDSPSwapChannels(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSwapChannels", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSwapChannels", "Swaps the left and right channels.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({}, "swap_channels")

    def process(self, audio):
        return (dsp.swap_channels(audio),)


class ComfyAudioDSPMonoMaker(_AudioDSPNode):
    CATEGORY = CATEGORY_STEREO
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMonoMaker", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMonoMaker", "Makes frequencies below a cutoff mono for low-frequency management.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "mono_maker"
        return cls._finish({
            "cutoff_hz": _float(section, "cutoff_hz", 140.0, 20.0, 1000.0, 1.0),
            "slope_order": _int(section, "slope_order", 4, 1, 12),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, cutoff_hz, slope_order, mix):
        return (dsp.mono_maker(audio, cutoff_hz, slope_order, mix),)


class ComfyAudioDSPBinauralPanner(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBinauralPanner", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPBinauralPanner", "Places a mono source in a simple binaural field using ITD/ILD cues.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "binaural_panner"
        return cls._finish({
            "azimuth_deg": _float(section, "azimuth_deg", 0.0, -180.0, 180.0, 1.0),
            "elevation_deg": _float(section, "elevation_deg", 0.0, -90.0, 90.0, 1.0),
            "distance_m": _float(section, "distance_m", 1.0, 0.1, 100.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, azimuth_deg, elevation_deg, distance_m, mix):
        return (dsp.binaural_panner(audio, azimuth_deg, elevation_deg, distance_m, mix),)


class ComfyAudioDSPHRTFConvolution(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHRTFConvolution", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHRTFConvolution", "Loads a SOFA HRTF file and renders the nearest binaural impulse response.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "hrtf_convolution"
        return cls._finish({
            "sofa_path": _string(section, "sofa_path"),
            "azimuth_deg": _float(section, "azimuth_deg", 0.0, -180.0, 180.0, 1.0),
            "elevation_deg": _float(section, "elevation_deg", 0.0, -90.0, 90.0, 1.0),
            "normalize_ir": _bool(section, "normalize_ir", True),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, sofa_path, azimuth_deg, elevation_deg, normalize_ir, mix):
        return (dsp.hrtf_convolution(audio, sofa_path, azimuth_deg, elevation_deg, normalize_ir, mix),)


class ComfyAudioDSPAmbisonicsEncoder(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAmbisonicsEncoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAmbisonicsEncoder", "Encodes mono audio to first-order Ambisonics WXYZ.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "ambisonics_encoder"
        return cls._finish({
            "azimuth_deg": _float(section, "azimuth_deg", 0.0, -180.0, 180.0, 1.0),
            "elevation_deg": _float(section, "elevation_deg", 0.0, -90.0, 90.0, 1.0),
            "gain_db": _float(section, "gain_db", 0.0, -24.0, 24.0, 0.1),
        }, section)

    def process(self, audio, azimuth_deg, elevation_deg, gain_db):
        return (dsp.ambisonics_encoder(audio, azimuth_deg, elevation_deg, gain_db),)


class ComfyAudioDSPAmbisonicsDecoder(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAmbisonicsDecoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAmbisonicsDecoder", "Decodes first-order Ambisonics to stereo or simple binaural output.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "ambisonics_decoder"
        return cls._finish({
            "mode": _combo(section, "mode", SPATIAL_DECODER_MODES, "stereo"),
            "width": _float(section, "width", 1.0, 0.0, 2.0, 0.01),
        }, section)

    def process(self, audio, mode, width):
        return (dsp.ambisonics_decoder(audio, mode, width),)


class ComfyAudioDSPAmbisonicsRotator(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAmbisonicsRotator", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAmbisonicsRotator", "Rotates a first-order Ambisonics WXYZ soundfield.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "ambisonics_rotator"
        return cls._finish({
            "yaw_deg": _float(section, "yaw_deg", 0.0, -180.0, 180.0, 1.0),
            "pitch_deg": _float(section, "pitch_deg", 0.0, -90.0, 90.0, 1.0),
            "roll_deg": _float(section, "roll_deg", 0.0, -180.0, 180.0, 1.0),
        }, section)

    def process(self, audio, yaw_deg, pitch_deg, roll_deg):
        return (dsp.ambisonics_rotator(audio, yaw_deg, pitch_deg, roll_deg),)


class ComfyAudioDSPDistanceSimulator(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDistanceSimulator", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDistanceSimulator", "Simulates distance with gain loss, air absorption, predelay, and reverb blend.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "distance_simulator"
        return cls._finish({
            "distance_m": _float(section, "distance_m", 5.0, 0.0, 200.0, 0.1),
            "air_absorption": _float(section, "air_absorption", 0.5, 0.0, 1.0, 0.01),
            "room_mix": _float(section, "room_mix", 0.35, 0.0, 1.0, 0.01),
            "dry_gain_db": _float(section, "dry_gain_db", 0.0, -24.0, 24.0, 0.1),
            "reverb_time_s": _float(section, "reverb_time_s", 1.5, 0.1, 20.0, 0.1),
        }, section)

    def process(self, audio, distance_m, air_absorption, room_mix, dry_gain_db, reverb_time_s):
        return (dsp.distance_simulator(audio, distance_m, air_absorption, room_mix, dry_gain_db, reverb_time_s),)


class ComfyAudioDSPDopplerEffect(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDopplerEffect", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDopplerEffect", "Simulates Doppler pitch and delay from changing source distance and velocity.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "doppler_effect"
        return cls._finish({
            "start_distance_m": _float(section, "start_distance_m", 20.0, 0.0, 500.0, 0.1),
            "end_distance_m": _float(section, "end_distance_m", 2.0, 0.0, 500.0, 0.1),
            "source_speed_m_s": _float(section, "source_speed_m_s", 0.0, -200.0, 200.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, start_distance_m, end_distance_m, source_speed_m_s, mix):
        return (dsp.doppler_effect(audio, start_distance_m, end_distance_m, source_speed_m_s, mix),)


class ComfyAudioDSPRMSMeter(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPRMSMeter", ("audio", "rms_dbfs", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPRMSMeter", "Reports RMS level in dBFS without changing the audio.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "rms_meter"
        return cls._finish({"window_ms": _float(section, "window_ms", 400.0, 1.0, 10000.0, 1.0)}, section)

    def process(self, audio, window_ms):
        return dsp.rms_meter(audio, window_ms)


class ComfyAudioDSPPeakMeter(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPeakMeter", ("audio", "peak_dbfs", "overload", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPPeakMeter", "Reports peak level and overload status.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "peak_meter"
        return cls._finish({"overload_db": _float(section, "overload_db", -0.1, -12.0, 6.0, 0.1)}, section)

    def process(self, audio, overload_db):
        return dsp.peak_meter(audio, overload_db)


class ComfyAudioDSPLUFSMeter(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLUFSMeter", ("audio", "integrated_lufs", "short_term_lufs", "momentary_lufs", "lra_lu", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPLUFSMeter", "Reports approximate integrated, short-term, momentary LUFS and loudness range.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({}, "lufs_meter")

    def process(self, audio):
        return dsp.lufs_meter(audio)


class ComfyAudioDSPLoudnessGraph:
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("IMAGE", "TENSOR")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLoudnessGraph", ("loudness_graph", "time_series"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPLoudnessGraph", "Plots an RMS or short-term LUFS time series and returns the underlying tensor.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "loudness_graph"
        return {"required": {
            "audio": _audio_input(section),
            "mode": _combo(section, "mode", ["rms_envelope", "short_term_lufs"], "rms_envelope"),
            "time_smoothing_s": _float(section, "time_smoothing_s", 0.4, 0.0, 10.0, 0.05),
            "color_scheme": _combo(section, "color_scheme", LOUDNESS_GRAPH_COLORS, "cyan"),
            "min_db": _float(section, "min_db", -60.0, -160.0, -1.0, 1.0),
            "max_db": _float(section, "max_db", 0.0, -60.0, 24.0, 1.0),
            "width": _int(section, "width", 960, 256, 4096, 16),
            "height": _int(section, "height", 420, 192, 2160, 16),
        }}

    def process(self, audio, mode, time_smoothing_s, color_scheme, min_db, max_db, width, height):
        if max_db <= min_db:
            raise ValueError("max_db must be greater than min_db")
        return dsp.loudness_graph(audio, mode, time_smoothing_s, color_scheme, min_db, max_db, width, height)


class ComfyAudioDSPSpectralAnalyzer(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectralAnalyzer", ("audio", "spectrum_image", "spectrum_data"))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectralAnalyzer", "Outputs an FFT spectrum image and sampled bin data.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectral_analyzer"
        return cls._finish({
            "fft_size": _int(section, "fft_size", 2048, 64, 32768, 64),
            "min_db": _float(section, "min_db", -100.0, -160.0, 0.0, 1.0),
            "max_db": _float(section, "max_db", 0.0, -80.0, 24.0, 1.0),
            "width": _int(section, "width", 768, 128, 2048, 16),
            "height": _int(section, "height", 384, 128, 2048, 16),
        }, section)

    def process(self, audio, fft_size, min_db, max_db, width, height):
        return dsp.spectral_analyzer(audio, fft_size, min_db, max_db, width, height)


class ComfyAudioDSPSpectrogramVisualizer(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectrogramVisualizer", ("audio", "spectrogram"))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectrogramVisualizer", "Generates a spectrogram image for preview.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectrogram_visualizer"
        return cls._finish({
            "fft_size": _int(section, "fft_size", 1024, 64, 8192, 64),
            "hop_size": _int(section, "hop_size", 256, 16, 4096, 16),
            "min_db": _float(section, "min_db", -100.0, -160.0, 0.0, 1.0),
            "max_db": _float(section, "max_db", 0.0, -80.0, 24.0, 1.0),
            "width": _int(section, "width", 768, 128, 2048, 16),
            "height": _int(section, "height", 384, 128, 2048, 16),
        }, section)

    def process(self, audio, fft_size, hop_size, min_db, max_db, width, height):
        return dsp.spectrogram_visualizer(audio, fft_size, hop_size, min_db, max_db, width, height)


class ComfyAudioDSPWaveformVisualizer(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPWaveformVisualizer", ("audio", "waveform_image"))
    DESCRIPTION = loc.description("ComfyAudioDSPWaveformVisualizer", "Generates a waveform preview image.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "waveform_visualizer"
        return cls._finish({
            "width": _int(section, "width", 768, 128, 2048, 16),
            "height": _int(section, "height", 320, 128, 2048, 16),
            "seconds": _float(section, "seconds", 0.0, 0.0, 600.0, 0.1),
        }, section)

    def process(self, audio, width, height, seconds):
        return dsp.waveform_visualizer(audio, width, height, seconds)


class ComfyAudioDSPPhaseCorrelationMeter(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPhaseCorrelationMeter", ("audio", "correlation", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPPhaseCorrelationMeter", "Reports stereo phase correlation from -1 to +1.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({}, "phase_correlation_meter")

    def process(self, audio):
        return dsp.phase_correlation_meter(audio)


class ComfyAudioDSPGoniometerVectorscope(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPGoniometerVectorscope", ("audio", "vectorscope"))
    DESCRIPTION = loc.description("ComfyAudioDSPGoniometerVectorscope", "Outputs a Lissajous vectorscope image for stereo phase.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "goniometer_vectorscope"
        return cls._finish({"width": _int(section, "width", 512, 128, 2048, 16), "height": _int(section, "height", 512, 128, 2048, 16)}, section)

    def process(self, audio, width, height):
        return dsp.goniometer_vectorscope(audio, width, height)


class ComfyAudioDSPBPMTempoDetector(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBPMTempoDetector", ("audio", "bpm", "beat_times"))
    DESCRIPTION = loc.description("ComfyAudioDSPBPMTempoDetector", "Estimates BPM and outputs beat positions as text.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "bpm_tempo_detector"
        return cls._finish({"min_bpm": _float(section, "min_bpm", 60.0, 20.0, 240.0, 1.0), "max_bpm": _float(section, "max_bpm", 180.0, 40.0, 320.0, 1.0)}, section)

    def process(self, audio, min_bpm, max_bpm):
        return dsp.bpm_tempo_detector(audio, min_bpm, max_bpm)


class ComfyAudioDSPKeyPitchDetector(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPKeyPitchDetector", ("audio", "key", "pitch_hz", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPKeyPitchDetector", "Estimates the dominant pitch and nearest pitch class.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "key_pitch_detector"
        return cls._finish({"min_hz": _float(section, "min_hz", 50.0, 10.0, 1000.0, 1.0), "max_hz": _float(section, "max_hz", 1600.0, 100.0, 8000.0, 1.0)}, section)

    def process(self, audio, min_hz, max_hz):
        return dsp.key_pitch_detector(audio, min_hz, max_hz)


class ComfyAudioDSPTransientOnsetDetector(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTransientOnsetDetector", ("audio", "onset_times", "count"))
    DESCRIPTION = loc.description("ComfyAudioDSPTransientOnsetDetector", "Detects transient/onset times and returns them as text.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "transient_onset_detector"
        return cls._finish({"sensitivity": _float(section, "sensitivity", 0.65, 0.0, 1.0, 0.01), "min_gap_ms": _float(section, "min_gap_ms", 80.0, 1.0, 2000.0, 1.0)}, section)

    def process(self, audio, sensitivity, min_gap_ms):
        return dsp.onset_detector(audio, sensitivity, min_gap_ms)


class ComfyAudioDSPSilenceDetector(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSilenceDetector", ("audio", "silence_ranges", "count"))
    DESCRIPTION = loc.description("ComfyAudioDSPSilenceDetector", "Detects silent time ranges below a threshold.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "silence_detector"
        return cls._finish({"threshold_db": _float(section, "threshold_db", -60.0, -120.0, 0.0, 1.0), "min_duration_ms": _float(section, "min_duration_ms", 250.0, 1.0, 60000.0, 1.0)}, section)

    def process(self, audio, threshold_db, min_duration_ms):
        return dsp.silence_detector(audio, threshold_db, min_duration_ms)


class ComfyAudioDSPSineWaveGenerator:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSineWaveGenerator", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPSineWaveGenerator", "Generates a sine wave test tone.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "sine_wave_generator"
        return {"required": {"frequency_hz": _float(section, "frequency_hz", 440.0, 0.01, 22000.0, 0.01), "amplitude": _float(section, "amplitude", 0.25, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 1.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "channels": _int(section, "channels", 2, 1, 8)}}

    def process(self, frequency_hz, amplitude, duration_s, sample_rate, channels):
        return (dsp.sine_wave_generator(frequency_hz, amplitude, duration_s, sample_rate, channels),)


class ComfyAudioDSPNoiseGenerator:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPNoiseGenerator", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPNoiseGenerator", "Generates white, pink, or brown noise.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "noise_generator"
        return {"required": {"noise_type": _combo(section, "noise_type", NOISE_TYPES, "white"), "amplitude": _float(section, "amplitude", 0.25, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 1.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "channels": _int(section, "channels", 2, 1, 8), "seed": _int(section, "seed", 0, 0, 2147483647)}}

    def process(self, noise_type, amplitude, duration_s, sample_rate, channels, seed):
        return (dsp.noise_generator(noise_type, amplitude, duration_s, sample_rate, channels, seed),)


class ComfyAudioDSPSweepChirp:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSweepChirp", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPSweepChirp", "Generates a linear or logarithmic sweep/chirp.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "sweep_chirp"
        return {"required": {"start_hz": _float(section, "start_hz", 20.0, 0.01, 22000.0, 0.01), "end_hz": _float(section, "end_hz", 20000.0, 0.01, 22000.0, 0.01), "amplitude": _float(section, "amplitude", 0.25, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 5.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "mode": _combo(section, "mode", SWEEP_MODES, "logarithmic"), "channels": _int(section, "channels", 2, 1, 8)}}

    def process(self, start_hz, end_hz, amplitude, duration_s, sample_rate, mode, channels):
        return (dsp.sweep_chirp(start_hz, end_hz, amplitude, duration_s, sample_rate, mode, channels),)


class ComfyAudioDSPImpulse:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPImpulse", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPImpulse", "Generates a single-sample impulse or short click.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "impulse"
        return {"required": {"amplitude": _float(section, "amplitude", 1.0, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 1.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "position_ms": _float(section, "position_ms", 0.0, 0.0, 3600000.0, 0.1), "click_ms": _float(section, "click_ms", 0.0, 0.0, 1000.0, 0.01), "channels": _int(section, "channels", 2, 1, 8)}}

    def process(self, amplitude, duration_s, sample_rate, position_ms, click_ms, channels):
        return (dsp.impulse(amplitude, duration_s, sample_rate, position_ms, click_ms, channels),)


class ComfyAudioDSPOscillatorMultiWave:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPOscillatorMultiWave", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPOscillatorMultiWave", "Generates sine, triangle, saw, or square oscillator audio.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "oscillator_multiwave"
        return {"required": {"waveform_shape": _combo(section, "waveform_shape", OSCILLATOR_WAVES, "sine"), "frequency_hz": _float(section, "frequency_hz", 440.0, 0.01, 22000.0, 0.01), "amplitude": _float(section, "amplitude", 0.25, 0.0, 1.0, 0.01), "duty_cycle": _float(section, "duty_cycle", 0.5, 0.01, 0.99, 0.01), "duration_s": _float(section, "duration_s", 1.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "channels": _int(section, "channels", 2, 1, 8)}}

    def process(self, waveform_shape, frequency_hz, amplitude, duty_cycle, duration_s, sample_rate, channels):
        return (dsp.oscillator_multiwave(waveform_shape, frequency_hz, amplitude, duty_cycle, duration_s, sample_rate, channels),)


class ComfyAudioDSPClickTrackMetronome:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPClickTrackMetronome", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPClickTrackMetronome", "Generates a click track/metronome by BPM and bar count.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "click_track_metronome"
        return {"required": {"bpm": _float(section, "bpm", 120.0, 20.0, 320.0, 0.1), "beats_per_bar": _int(section, "beats_per_bar", 4, 1, 16), "bars": _int(section, "bars", 4, 1, 1024), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "accent_frequency_hz": _float(section, "accent_frequency_hz", 1600.0, 20.0, 20000.0, 1.0), "beat_frequency_hz": _float(section, "beat_frequency_hz", 1000.0, 20.0, 20000.0, 1.0), "amplitude": _float(section, "amplitude", 0.5, 0.0, 1.0, 0.01)}, "optional": {"song_bpm": _force_input(section, "song_bpm", "FLOAT"), "analysis_json": _force_input(section, "analysis_json", "STRING")}}

    def process(self, bpm, beats_per_bar, bars, sample_rate, accent_frequency_hz, beat_frequency_hz, amplitude, song_bpm=None, analysis_json=None):
        bpm = dsp.song_bpm_value(bpm, song_bpm, analysis_json)
        return (dsp.click_track_metronome(bpm, beats_per_bar, bars, sample_rate, accent_frequency_hz, beat_frequency_hz, amplitude),)


class ComfyAudioDSPAudioMixer(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioMixer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioMixer", "Mixes up to eight audio tracks with gain, pan, mute, solo, and master gain.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "audio_mixer"
        required = {"audio_1": _audio_input(section, "audio_1"), "master_gain_db": _float(section, "master_gain_db", 0.0, -60.0, 24.0, 0.1)}
        optional = {}
        for index in range(2, 9):
            optional[f"audio_{index}"] = _optional_audio(section, f"audio_{index}")
        for index in range(1, 9):
            required[f"track_{index}_gain_db"] = _float(section, f"track_{index}_gain_db", 0.0, -60.0, 24.0, 0.1)
            required[f"track_{index}_pan"] = _float(section, f"track_{index}_pan", 0.0, -1.0, 1.0, 0.01)
            required[f"track_{index}_mute"] = _bool(section, f"track_{index}_mute", False)
            required[f"track_{index}_solo"] = _bool(section, f"track_{index}_solo", False)
        return {"required": required, "optional": optional}

    def process(self, audio_1, master_gain_db, **kwargs):
        tracks = []
        for index in range(1, 9):
            tracks.append((audio_1 if index == 1 else kwargs.get(f"audio_{index}"), kwargs[f"track_{index}_gain_db"], kwargs[f"track_{index}_pan"], kwargs[f"track_{index}_mute"], kwargs[f"track_{index}_solo"]))
        return (dsp.audio_mixer(audio_1, tracks, master_gain_db),)


class ComfyAudioDSPAudioRouterSelector(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioRouterSelector", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioRouterSelector", "Selects one of up to four audio inputs by index.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "audio_router_selector"
        return {"required": {"audio_1": _audio_input(section, "audio_1"), "index": _int(section, "index", 1, 1, 4)}, "optional": {"audio_2": _optional_audio(section, "audio_2"), "audio_3": _optional_audio(section, "audio_3"), "audio_4": _optional_audio(section, "audio_4")}}

    def process(self, audio_1, index, audio_2=None, audio_3=None, audio_4=None):
        return (dsp.audio_selector(index, [audio_1, audio_2, audio_3, audio_4]),)


class ComfyAudioDSPAudioSplitter(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioSplitter", ("channel_1", "channel_2", "channel_3", "channel_4"))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioSplitter", "Splits stereo or multichannel audio into mono channel outputs.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({}, "audio_splitter")

    def process(self, audio):
        return dsp.audio_splitter(audio)


class ComfyAudioDSPAudioMerger(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioMerger", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioMerger", "Merges mono inputs into stereo or multichannel audio.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "audio_merger"
        return {"required": {"audio_1": _audio_input(section, "audio_1"), "output_mode": _combo(section, "output_mode", ["stereo", "multichannel"], "stereo")}, "optional": {f"audio_{index}": _optional_audio(section, f"audio_{index}") for index in range(2, 9)}}

    def process(self, audio_1, output_mode, **kwargs):
        return (dsp.audio_merger(audio_1, [kwargs.get(f"audio_{index}") for index in range(2, 9)], output_mode),)


class ComfyAudioDSPCrossfader(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPCrossfader", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPCrossfader", "Crossfades between two audio inputs.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "crossfader"
        return {"required": {"audio_a": _audio_input(section, "audio_a"), "audio_b": _audio_input(section, "audio_b"), "fade": _float(section, "fade", 0.5, 0.0, 1.0, 0.01), "equal_power": _bool(section, "equal_power", True)}}

    def process(self, audio_a, audio_b, fade, equal_power):
        return (dsp.crossfader(audio_a, audio_b, fade, equal_power),)


class ComfyAudioDSPSidechainGateCompressor(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSidechainGateCompressor", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSidechainGateCompressor", "Applies gate or compressor gain to audio using an external sidechain key.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "sidechain_gate_compressor"
        return {"required": {"audio": _audio_input(section), "sidechain": _audio_input(section, "sidechain"), "mode": _combo(section, "mode", ["compressor", "gate"], "compressor"), "threshold_db": _float(section, "threshold_db", -24.0, -100.0, 0.0), "ratio": _float(section, "ratio", 4.0, 1.0, 50.0), "attack_ms": _float(section, "attack_ms", 10.0, 0.0, 500.0), "release_ms": _float(section, "release_ms", 120.0, 1.0, 5000.0, 1.0), "range_db": _float(section, "range_db", 60.0, 0.0, 120.0, 0.5), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}}

    def process(self, audio, sidechain, mode, threshold_db, ratio, attack_ms, release_ms, range_db, mix):
        return (dsp.sidechain_gate_compressor(audio, sidechain, mode, threshold_db, ratio, attack_ms, release_ms, range_db, mix),)


class ComfyAudioDSPSendReturnLoop(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSendReturnLoop", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSendReturnLoop", "Blends dry audio with a processed return signal for send/return workflows.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "send_return_loop"
        return {"required": {"audio": _audio_input(section), "return_audio": _audio_input(section, "return_audio"), "send_level_db": _float(section, "send_level_db", 0.0, -60.0, 24.0, 0.1), "return_level_db": _float(section, "return_level_db", 0.0, -60.0, 24.0, 0.1), "dry_level_db": _float(section, "dry_level_db", 0.0, -60.0, 24.0, 0.1)}}

    def process(self, audio, return_audio, send_level_db, return_level_db, dry_level_db):
        return (dsp.send_return_loop(audio, return_audio, send_level_db, return_level_db, dry_level_db),)


class ComfyAudioDSPParallelProcessingRouter(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPParallelProcessingRouter", ("dry_audio", "send_audio"))
    DESCRIPTION = loc.description("ComfyAudioDSPParallelProcessingRouter", "Splits audio into dry and level-adjusted send paths for parallel processing.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "parallel_processing_router"
        return cls._finish({"send_level_db": _float(section, "send_level_db", 0.0, -60.0, 24.0, 0.1), "dry_level_db": _float(section, "dry_level_db", 0.0, -60.0, 24.0, 0.1)}, section)

    def process(self, audio, send_level_db, dry_level_db):
        return dsp.parallel_processing_router(audio, send_level_db, dry_level_db)


class ComfyAudioDSPParallelReturnMixer(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_NAMES = loc.return_names("ComfyAudioDSPParallelReturnMixer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPParallelReturnMixer", "Mixes a dry path with a processed parallel return.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "parallel_return_mixer"
        return {"required": {"dry_audio": _audio_input(section, "dry_audio"), "processed_audio": _audio_input(section, "processed_audio"), "processed_level_db": _float(section, "processed_level_db", 0.0, -60.0, 24.0, 0.1), "dry_level_db": _float(section, "dry_level_db", 0.0, -60.0, 24.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}}

    def process(self, dry_audio, processed_audio, processed_level_db, dry_level_db, mix):
        return (dsp.parallel_return_mixer(dry_audio, processed_audio, processed_level_db, dry_level_db, mix),)


class ComfyAudioDSPGainTrim(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPGainTrim", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPGainTrim", "Applies simple gain trim in dB.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({"gain_db": _float("gain_trim", "gain_db", 0.0, -120.0, 60.0, 0.1)}, "gain_trim")

    def process(self, audio, gain_db):
        return (dsp.gain_trim(audio, gain_db),)


class ComfyAudioDSPPhaseInverter(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPhaseInverter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPhaseInverter", "Inverts polarity on mono, left, and/or right channels.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "phase_inverter"
        return cls._finish({"invert_left": _bool(section, "invert_left", True), "invert_right": _bool(section, "invert_right", True)}, section)

    def process(self, audio, invert_left, invert_right):
        return (dsp.phase_inverter(audio, invert_left, invert_right),)


class ComfyAudioDSPDCOffsetRemover(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDCOffsetRemover", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDCOffsetRemover", "Removes DC offset with mean removal or high-pass filtering.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({"highpass_hz": _float("dc_offset_remover", "highpass_hz", 20.0, 0.0, 200.0, 0.1)}, "dc_offset_remover")

    def process(self, audio, highpass_hz):
        return (dsp.dc_offset_remover(audio, highpass_hz),)


class ComfyAudioDSPFadeInOut(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFadeInOut", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFadeInOut", "Applies fade in and fade out with selectable curve shape.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "fade_in_out"
        return cls._finish({"fade_in_ms": _float(section, "fade_in_ms", 50.0, 0.0, 3600000.0, 1.0), "fade_out_ms": _float(section, "fade_out_ms", 50.0, 0.0, 3600000.0, 1.0), "curve": _combo(section, "curve", FADE_CURVES, "linear")}, section)

    def process(self, audio, fade_in_ms, fade_out_ms, curve):
        return (dsp.fade_in_out(audio, fade_in_ms, fade_out_ms, curve),)


class ComfyAudioDSPAudioTrimCrop(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioTrimCrop", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioTrimCrop", "Crops audio between start and end time.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "audio_trim_crop"
        return cls._finish({"start_s": _float(section, "start_s", 0.0, 0.0, 36000.0, 0.001), "end_s": _float(section, "end_s", 0.0, 0.0, 36000.0, 0.001)}, section)

    def process(self, audio, start_s, end_s):
        return (dsp.audio_trim_crop(audio, start_s, end_s),)


class ComfyAudioDSPSilenceTrimmer(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSilenceTrimmer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSilenceTrimmer", "Automatically trims leading and trailing silence.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "silence_trimmer"
        return cls._finish({"threshold_db": _float(section, "threshold_db", -60.0, -120.0, 0.0, 1.0), "padding_ms": _float(section, "padding_ms", 20.0, 0.0, 10000.0, 1.0)}, section)

    def process(self, audio, threshold_db, padding_ms):
        return (dsp.silence_trimmer(audio, threshold_db, padding_ms),)


class ComfyAudioDSPNormalize(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPNormalize", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPNormalize", "Normalizes audio by peak, RMS, or approximate LUFS target.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "normalize"
        return cls._finish({"mode": _combo(section, "mode", NORMALIZE_MODES, "peak"), "target_db": _float(section, "target_db", -1.0, -80.0, 24.0, 0.1)}, section)

    def process(self, audio, mode, target_db):
        return (dsp.normalize_audio(audio, mode, target_db),)


class ComfyAudioDSPResampleChangeSampleRate(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPResampleChangeSampleRate", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPResampleChangeSampleRate", "Resamples audio to a target sample rate.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({"target_sample_rate": _int("resample_change_sample_rate", "target_sample_rate", 44100, 8000, 384000, 1)}, "resample_change_sample_rate")

    def process(self, audio, target_sample_rate):
        return (dsp.resample_change_sample_rate(audio, target_sample_rate),)


class ComfyAudioDSPFormatConverter(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFormatConverter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFormatConverter", "Converts between mono and stereo channel formats.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({"mode": _combo("format_converter", "mode", FORMAT_MODES, "stereo_duplicate")}, "format_converter")

    def process(self, audio, mode):
        return (dsp.format_converter(audio, mode),)


class ComfyAudioDSPAudioInfo(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_TYPES = ("AUDIO", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioInfo", ("audio", "sample_rate", "duration_s", "channels", "samples", "info"))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioInfo", "Outputs sample rate, duration, channel count, sample count, and metadata text.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({}, "audio_info")

    def process(self, audio):
        return dsp.audio_info(audio)


class ComfyAudioDSPDelayCompensation(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDelayCompensation", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDelayCompensation", "Adds manual sample or millisecond delay for alignment.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "delay_compensation"
        return cls._finish({"delay_samples": _int(section, "delay_samples", 0, 0, 10000000, 1), "delay_ms": _float(section, "delay_ms", 0.0, 0.0, 3600000.0, 0.01)}, section)

    def process(self, audio, delay_samples, delay_ms):
        return (dsp.delay_compensation(audio, delay_samples, delay_ms),)


class ComfyAudioDSPLoopDuplicator(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLoopDuplicator", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPLoopDuplicator", "Loops audio a fixed number of times or to a target duration.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "loop_duplicator"
        return cls._finish({"loops": _int(section, "loops", 2, 1, 10000, 1), "target_duration_s": _float(section, "target_duration_s", 0.0, 0.0, 36000.0, 0.001)}, section)

    def process(self, audio, loops, target_duration_s):
        return (dsp.loop_duplicator(audio, loops, target_duration_s),)


class ComfyAudioDSPReverseAudio(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPReverseAudio", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPReverseAudio", "Reverses audio playback direction.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({}, "reverse_audio")

    def process(self, audio):
        return (dsp.reverse_audio(audio),)


class ComfyAudioDSPDither(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDither", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDither", "Applies RPDF, TPDF, or simple noise-shaped dither before fixed-point quantization.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "dither"
        return cls._finish({"bit_depth": _int(section, "bit_depth", 24, 8, 32, 1), "dither_type": _combo(section, "dither_type", DITHER_TYPES, "tpdf"), "noise_shape": _float(section, "noise_shape", 0.5, 0.0, 0.95, 0.01), "seed": _int(section, "seed", 0, 0, 2147483647, 1)}, section)

    def process(self, audio, bit_depth, dither_type, noise_shape, seed):
        return (dsp.dither(audio, bit_depth, dither_type, noise_shape, seed),)


class ComfyAudioDSPDynamicEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDynamicEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDynamicEQ", "Single-band dynamic EQ whose band gain follows input level.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "dynamic_eq"
        return cls._finish({
            "frequency_hz": _float(section, "frequency_hz", 1000.0, 20.0, 22000.0, 1.0),
            "q": _float(section, "q", 2.0, 0.1, 30.0, 0.01),
            "threshold_db": _float(section, "threshold_db", -24.0, -100.0, 0.0),
            "ratio": _float(section, "ratio", 4.0, 1.0, 50.0),
            "attack_ms": _float(section, "attack_ms", 10.0, 0.0, 500.0),
            "release_ms": _float(section, "release_ms", 120.0, 1.0, 5000.0, 1.0),
            "range_db": _float(section, "range_db", 12.0, 0.0, 48.0, 0.1),
            "mode": _combo(section, "mode", ["cut_above", "boost_above", "cut_below", "boost_below"], "cut_above"),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, frequency_hz, q, threshold_db, ratio, attack_ms, release_ms, range_db, mode, mix):
        return (dsp.dynamic_eq(audio, frequency_hz, q, threshold_db, ratio, attack_ms, release_ms, range_db, mode, mix),)


class ComfyAudioDSPVocoder(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPVocoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPVocoder", "Filter-bank vocoder using a modulator envelope to shape a carrier.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "vocoder"
        return {"required": {
            "carrier": _audio_input(section, "carrier"),
            "modulator": _audio_input(section, "modulator"),
            "bands": _int(section, "bands", 16, 4, 32),
            "low_hz": _float(section, "low_hz", 80.0, 20.0, 8000.0, 1.0),
            "high_hz": _float(section, "high_hz", 8000.0, 200.0, 22000.0, 1.0),
            "attack_ms": _float(section, "attack_ms", 5.0, 0.0, 200.0),
            "release_ms": _float(section, "release_ms", 80.0, 1.0, 2000.0, 1.0),
            "modulator_gain_db": _float(section, "modulator_gain_db", 18.0, -24.0, 48.0, 0.1),
            "carrier_gain_db": _float(section, "carrier_gain_db", 0.0, -24.0, 24.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }}

    def process(self, carrier, modulator, bands, low_hz, high_hz, attack_ms, release_ms, modulator_gain_db, carrier_gain_db, mix):
        return (dsp.vocoder(carrier, modulator, bands, low_hz, high_hz, attack_ms, release_ms, modulator_gain_db, carrier_gain_db, mix),)


class ComfyAudioDSPEnvelopeFollowerOutput(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "STRING", "AUDIO")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPEnvelopeFollowerOutput", ("audio", "current", "average", "envelope_points", "envelope_audio"))
    DESCRIPTION = loc.description("ComfyAudioDSPEnvelopeFollowerOutput", "Extracts an attack/release-smoothed envelope as scalar, text, and audio-rate control signal.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "envelope_follower_output"
        return cls._finish({
            "attack_ms": _float(section, "attack_ms", 10.0, 0.0, 5000.0),
            "release_ms": _float(section, "release_ms", 120.0, 1.0, 10000.0, 1.0),
            "mode": _combo(section, "mode", ["rms", "peak"], "rms"),
            "normalize": _bool(section, "normalize", True),
            "points": _int(section, "points", 128, 2, 4096),
        }, section)

    def process(self, audio, attack_ms, release_ms, mode, normalize, points):
        return dsp.envelope_follower_output(audio, attack_ms, release_ms, mode, normalize, points)


class ComfyAudioDSPMultibandCrossover(_AudioDSPNode):
    CATEGORY = CATEGORY_ROUTING
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMultibandCrossover", ("low", "low_mid", "high_mid", "high"))
    DESCRIPTION = loc.description("ComfyAudioDSPMultibandCrossover", "Splits audio into three or four frequency bands for parallel processing.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "multiband_crossover"
        return cls._finish({
            "bands": _combo(section, "bands", ["3", "4"], "3"),
            "crossover_low_hz": _float(section, "crossover_low_hz", 160.0, 20.0, 22000.0, 1.0),
            "crossover_mid_hz": _float(section, "crossover_mid_hz", 1200.0, 20.0, 22000.0, 1.0),
            "crossover_high_hz": _float(section, "crossover_high_hz", 6000.0, 20.0, 22000.0, 1.0),
        }, section)

    def process(self, audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz):
        return dsp.multiband_crossover(audio, bands, crossover_low_hz, crossover_mid_hz, crossover_high_hz)


class ComfyAudioDSPDeclickDecrackle(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDeclickDecrackle", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDeclickDecrackle", "Detects short impulse clicks/crackle and repairs them with median interpolation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "declick_decrackle"
        return cls._finish({
            "threshold": _float(section, "threshold", 8.0, 1.0, 40.0, 0.1),
            "window_samples": _int(section, "window_samples", 9, 3, 101, 2),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, threshold, window_samples, mix):
        return (dsp.declick_decrackle(audio, threshold, window_samples, mix),)


class ComfyAudioDSPSpectralSmoothingContrast(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectralSmoothingContrast", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectralSmoothingContrast", "Smooths or contrast-enhances the spectral envelope, like audio unsharp masking.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectral_smoothing_contrast"
        return cls._finish({
            "mode": _combo(section, "mode", SPECTRAL_SHAPER_MODES, "smooth"),
            "amount": _float(section, "amount", 0.5, 0.0, 3.0, 0.01),
            "frequency_smoothing_bins": _int(section, "frequency_smoothing_bins", 9, 1, 129, 2),
            "fft_size": _int(section, "fft_size", 2048, 64, 32768, 64),
            "hop_size": _int(section, "hop_size", 512, 16, 8192, 16),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, mode, amount, frequency_smoothing_bins, fft_size, hop_size, mix):
        return (dsp.spectral_smoothing_contrast(audio, mode, amount, frequency_smoothing_bins, fft_size, hop_size, mix),)


class ComfyAudioDSPHumRemover(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHumRemover", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHumRemover", "Detects or selects 50/60 Hz mains hum and applies harmonic notch filtering.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "hum_remover"
        return cls._finish({
            "base_mode": _combo(section, "base_mode", HUM_BASE_MODES, "auto"),
            "max_harmonics": _int(section, "max_harmonics", 8, 1, 40),
            "q": _float(section, "q", 35.0, 1.0, 500.0, 0.1),
            "reduction_db": _float(section, "reduction_db", 48.0, 0.0, 80.0, 0.1),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, base_mode, max_harmonics, q, reduction_db, mix):
        return (dsp.hum_remover(audio, base_mode, max_harmonics, q, reduction_db, mix),)


class ComfyAudioDSPPhaseRotatorAllpass(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPhaseRotatorAllpass", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPhaseRotatorAllpass", "Single adjustable all-pass phase rotator without intentional magnitude change.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "phase_rotator_allpass"
        return cls._finish({
            "frequency_hz": _float(section, "frequency_hz", 1000.0, 20.0, 22000.0, 1.0),
            "q": _float(section, "q", 0.707, 0.05, 30.0, 0.001),
            "stages": _int(section, "stages", 1, 1, 12),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, frequency_hz, q, stages, mix):
        return (dsp.phase_rotator_allpass(audio, frequency_hz, q, stages, mix),)


class ComfyAudioDSPGranularProcessor(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPGranularProcessor", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPGranularProcessor", "Granular processor that replays windowed grains with pitch, jitter, scatter, and reverse probability.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "granular_processor"
        return cls._finish({
            "grain_ms": _float(section, "grain_ms", 80.0, 1.0, 1000.0, 0.1),
            "overlap": _float(section, "overlap", 2.0, 1.0, 16.0, 0.1),
            "pitch_semitones": _float(section, "pitch_semitones", 0.0, -24.0, 24.0, 0.01),
            "position_jitter_ms": _float(section, "position_jitter_ms", 20.0, 0.0, 1000.0, 0.1),
            "time_scatter": _float(section, "time_scatter", 0.25, 0.0, 1.0, 0.01),
            "reverse_probability": _float(section, "reverse_probability", 0.0, 0.0, 1.0, 0.01),
            "seed": _int(section, "seed", 0, 0, 2147483647),
            "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01),
        }, section)

    def process(self, audio, grain_ms, overlap, pitch_semitones, position_jitter_ms, time_scatter, reverse_probability, seed, mix):
        return (dsp.granular_processor(audio, grain_ms, overlap, pitch_semitones, position_jitter_ms, time_scatter, reverse_probability, seed, mix),)


class ComfyAudioDSPMathSignalMixer(_AudioDSPNode):
    CATEGORY = CATEGORY_UTILITIES
    RETURN_NAMES = loc.return_names("ComfyAudioDSPMathSignalMixer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPMathSignalMixer", "Safely evaluates a sample-wise formula over up to four audio inputs A-D.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "math_signal_mixer"
        return {"required": {"audio_a": _audio_input(section, "audio_a"), "expression": _string(section, "expression", "A", multiline=True), "gain_db": _float(section, "gain_db", 0.0, -60.0, 24.0, 0.1)}, "optional": {"audio_b": _optional_audio(section, "audio_b"), "audio_c": _optional_audio(section, "audio_c"), "audio_d": _optional_audio(section, "audio_d")}}

    def process(self, audio_a, expression, gain_db, audio_b=None, audio_c=None, audio_d=None):
        return (dsp.math_signal_mixer(audio_a, audio_b, audio_c, audio_d, expression, gain_db),)


class ComfyAudioDSPUpwardCompressor(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPUpwardCompressor", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPUpwardCompressor", "Raises low-level material below a threshold.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "upward_compressor"
        return cls._finish({"threshold_db": _float(section, "threshold_db", -36.0, -100.0, 0.0), "ratio": _float(section, "ratio", 2.0, 1.0, 20.0), "attack_ms": _float(section, "attack_ms", 20.0, 0.0, 1000.0), "release_ms": _float(section, "release_ms", 250.0, 1.0, 5000.0, 1.0), "range_db": _float(section, "range_db", 18.0, 0.0, 60.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, range_db, mix):
        return (dsp.upward_compressor(audio, threshold_db, ratio, attack_ms, release_ms, range_db, mix),)


class ComfyAudioDSPParallelCompressionMix(_AudioDSPNode):
    CATEGORY = CATEGORY_DYNAMICS
    RETURN_NAMES = loc.return_names("ComfyAudioDSPParallelCompressionMix", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPParallelCompressionMix", "Blends dry audio with a compressed parallel path.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "parallel_compression_mix"
        return cls._finish({"threshold_db": _float(section, "threshold_db", -24.0, -80.0, 0.0), "ratio": _float(section, "ratio", 8.0, 1.0, 50.0), "attack_ms": _float(section, "attack_ms", 5.0, 0.0, 500.0), "release_ms": _float(section, "release_ms", 120.0, 1.0, 5000.0, 1.0), "knee_db": _float(section, "knee_db", 6.0, 0.0, 48.0), "compressed_gain_db": _float(section, "compressed_gain_db", 6.0, -24.0, 48.0, 0.1), "dry_gain_db": _float(section, "dry_gain_db", 0.0, -24.0, 24.0, 0.1), "blend": _float(section, "blend", 0.5, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, threshold_db, ratio, attack_ms, release_ms, knee_db, compressed_gain_db, dry_gain_db, blend):
        return (dsp.parallel_compression_mix(audio, threshold_db, ratio, attack_ms, release_ms, knee_db, compressed_gain_db, dry_gain_db, blend),)


class ComfyAudioDSPLinearPhaseEQ(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLinearPhaseEQ", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPLinearPhaseEQ", "Zero-phase FFT EQ for linear-phase-style magnitude shaping.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "linear_phase_eq"
        return cls._finish({"band_type": _combo(section, "band_type", LINEAR_PHASE_EQ_TYPES, "peak"), "frequency_hz": _float(section, "frequency_hz", 1000.0, 20.0, 22000.0, 1.0), "gain_db": _float(section, "gain_db", 0.0, -48.0, 48.0, 0.1), "q": _float(section, "q", 1.0, 0.1, 30.0, 0.01), "slope": _float(section, "slope", 120.0, 1.0, 5000.0, 1.0), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, band_type, frequency_hz, gain_db, q, slope, mix):
        return (dsp.linear_phase_eq(audio, band_type, frequency_hz, gain_db, q, slope, mix),)


class ComfyAudioDSPCombFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_EQ
    RETURN_NAMES = loc.return_names("ComfyAudioDSPCombFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPCombFilter", "Feedforward/feedback comb filter for coloration and resonances.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "comb_filter"
        return cls._finish({"delay_ms": _float(section, "delay_ms", 5.0, 0.05, 100.0, 0.01), "feedback": _float(section, "feedback", 0.35, -0.98, 0.98, 0.01), "feedforward": _float(section, "feedforward", 0.5, -2.0, 2.0, 0.01), "damping_hz": _float(section, "damping_hz", 8000.0, 20.0, 22000.0, 1.0), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, delay_ms, feedback, feedforward, damping_hz, mix):
        return (dsp.comb_filter(audio, delay_ms, feedback, feedforward, damping_hz, mix),)


class ComfyAudioDSPShimmerReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPShimmerReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPShimmerReverb", "Reverb with octave-shifted shimmer mixed into the tail.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "shimmer_reverb"
        return cls._finish({"reverb_time_s": _float(section, "reverb_time_s", 3.0, 0.1, 30.0), "diffusion": _float(section, "diffusion", 0.8, 0.0, 1.0, 0.01), "shimmer_octaves": _float(section, "shimmer_octaves", 1.0, -2.0, 2.0, 0.01), "shimmer_amount": _float(section, "shimmer_amount", 0.5, 0.0, 2.0, 0.01), "high_cut_hz": _float(section, "high_cut_hz", 12000.0, 500.0, 22000.0, 1.0), "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, reverb_time_s, diffusion, shimmer_octaves, shimmer_amount, high_cut_hz, mix):
        return (dsp.shimmer_reverb(audio, reverb_time_s, diffusion, shimmer_octaves, shimmer_amount, high_cut_hz, mix),)


class ComfyAudioDSPFDNReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_REVERB
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFDNReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFDNReverb", "Feedback delay network reverb with diffusion, damping, and modulation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "fdn_reverb"
        return cls._finish({"decay_time_s": _float(section, "decay_time_s", 2.5, 0.1, 60.0, 0.01), "diffusion": _float(section, "diffusion", 0.75, 0.0, 1.0, 0.01), "damping": _float(section, "damping", 0.35, 0.0, 1.0, 0.01), "modulation": _float(section, "modulation", 0.2, 0.0, 1.0, 0.01), "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, decay_time_s, diffusion, damping, modulation, mix):
        return (dsp.fdn_reverb(audio, decay_time_s, diffusion, damping, modulation, mix),)


class ComfyAudioDSPReverseDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPReverseDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPReverseDelay", "Reverse-style delay by delaying reversed audio then flipping it back.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "reverse_delay"
        return cls._finish({"delay_ms": _float(section, "delay_ms", 350.0, 1.0, 5000.0, 1.0), "feedback": _float(section, "feedback", 0.25, -0.95, 0.95, 0.01), "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, delay_ms, feedback, mix):
        return (dsp.reverse_delay(audio, delay_ms, feedback, mix),)


class ComfyAudioDSPGranularDelay(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPGranularDelay", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPGranularDelay", "Delay whose wet path is broken into jittered grains.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "granular_delay"
        return cls._finish({"delay_ms": _float(section, "delay_ms", 300.0, 1.0, 5000.0, 1.0), "grain_ms": _float(section, "grain_ms", 60.0, 1.0, 1000.0, 0.1), "density": _float(section, "density", 2.0, 1.0, 16.0, 0.1), "pitch_semitones": _float(section, "pitch_semitones", 0.0, -24.0, 24.0, 0.01), "feedback": _float(section, "feedback", 0.25, -0.95, 0.95, 0.01), "seed": _int(section, "seed", 0, 0, 2147483647), "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, delay_ms, grain_ms, density, pitch_semitones, feedback, seed, mix):
        return (dsp.granular_delay(audio, delay_ms, grain_ms, density, pitch_semitones, feedback, seed, mix),)


class ComfyAudioDSPSlapEcho(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSlapEcho", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSlapEcho", "Preset slapback echo styles.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "slap_echo"
        return cls._finish({"style": _combo(section, "style", ["classic", "rockabilly", "wide"], "classic"), "mix": _float(section, "mix", 0.25, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, style, mix):
        return (dsp.slap_echo(audio, style, mix),)


class ComfyAudioDSPEchoplexTapeEcho(_AudioDSPNode):
    CATEGORY = CATEGORY_DELAY
    RETURN_NAMES = loc.return_names("ComfyAudioDSPEchoplexTapeEcho", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPEchoplexTapeEcho", "Echoplex-style tape echo with saturation, tone aging, wow, and feedback.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "echoplex_tape_echo"
        return cls._finish({"delay_ms": _float(section, "delay_ms", 260.0, 20.0, 2000.0, 1.0), "feedback": _float(section, "feedback", 0.35, 0.0, 0.95, 0.01), "tape_age": _float(section, "tape_age", 0.35, 0.0, 1.0, 0.01), "wow_depth_ms": _float(section, "wow_depth_ms", 1.5, 0.0, 20.0, 0.01), "record_level_db": _float(section, "record_level_db", 3.0, -24.0, 24.0, 0.1), "mix": _float(section, "mix", 0.35, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, delay_ms, feedback, tape_age, wow_depth_ms, record_level_db, mix):
        return (dsp.echoplex_tape_echo(audio, delay_ms, feedback, tape_age, wow_depth_ms, record_level_db, mix),)


class ComfyAudioDSPBarberpoleFlanger(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBarberpoleFlanger", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPBarberpoleFlanger", "Continuously rising or falling flanger illusion.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "barberpole_flanger"
        return cls._finish({"base_delay_ms": _float(section, "base_delay_ms", 1.0, 0.1, 20.0, 0.01), "depth_ms": _float(section, "depth_ms", 6.0, 0.0, 30.0, 0.01), "rate_hz": _float(section, "rate_hz", 0.15, 0.001, 5.0, 0.001), "feedback": _float(section, "feedback", 0.25, -0.95, 0.95, 0.01), "direction": _combo(section, "direction", ["up", "down"], "up"), "mix": _float(section, "mix", 0.5, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, base_delay_ms, depth_ms, rate_hz, feedback, direction, mix):
        return (dsp.barberpole_flanger(audio, base_delay_ms, depth_ms, rate_hz, feedback, direction, mix),)


class ComfyAudioDSPAutoFilter(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAutoFilter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAutoFilter", "LFO-controlled filter sweep.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "auto_filter"
        return cls._finish({"filter_type": _combo(section, "filter_type", AUTO_FILTER_TYPES, "low_pass"), "base_cutoff_hz": _float(section, "base_cutoff_hz", 1000.0, 20.0, 22000.0, 1.0), "depth_octaves": _float(section, "depth_octaves", 2.0, 0.0, 8.0, 0.01), "rate_hz": _float(section, "rate_hz", 0.5, 0.001, 30.0, 0.001), "resonance_q": _float(section, "resonance_q", 1.0, 0.1, 10.0, 0.01), "waveform_shape": _combo(section, "waveform_shape", WAVEFORMS, "sine"), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, filter_type, base_cutoff_hz, depth_octaves, rate_hz, resonance_q, waveform_shape, mix):
        return (dsp.auto_filter(audio, filter_type, base_cutoff_hz, depth_octaves, rate_hz, resonance_q, waveform_shape, mix),)


class ComfyAudioDSPAutoWah(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAutoWah", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAutoWah", "Envelope-controlled resonant band-pass wah filter.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "auto_wah"
        return cls._finish({"min_frequency_hz": _float(section, "min_frequency_hz", 350.0, 20.0, 12000.0, 1.0), "max_frequency_hz": _float(section, "max_frequency_hz", 2400.0, 40.0, 22000.0, 1.0), "q": _float(section, "q", 4.0, 0.1, 30.0, 0.01), "attack_ms": _float(section, "attack_ms", 8.0, 0.0, 1000.0, 0.1), "release_ms": _float(section, "release_ms", 180.0, 1.0, 5000.0, 1.0), "drive_db": _float(section, "drive_db", 6.0, -24.0, 36.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, min_frequency_hz, max_frequency_hz, q, attack_ms, release_ms, drive_db, mix):
        return (dsp.auto_wah(audio, min_frequency_hz, max_frequency_hz, q, attack_ms, release_ms, drive_db, mix),)


class ComfyAudioDSPRhythmicGateStutter(_AudioDSPNode):
    CATEGORY = CATEGORY_MODULATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPRhythmicGateStutter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPRhythmicGateStutter", "Beat-synced rhythmic gate and stutter pattern sequencer.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "rhythmic_gate_stutter"
        return cls._finish({"bpm": _float(section, "bpm", 120.0, 1.0, 400.0, 0.1), "division": _combo(section, "division", STUTTER_DIVISIONS, "1/16"), "pattern": _string(section, "pattern", "1,0,1,0,1,1,0,0"), "depth": _float(section, "depth", 1.0, 0.0, 1.0, 0.01), "smoothing_ms": _float(section, "smoothing_ms", 2.0, 0.0, 200.0, 0.1), "mode": _combo(section, "mode", ["gate", "stutter"], "gate"), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section, {"song_bpm": _force_input(section, "song_bpm", "FLOAT"), "analysis_json": _force_input(section, "analysis_json", "STRING")})

    def process(self, audio, bpm, division, pattern, depth, smoothing_ms, mode, mix, song_bpm=None, analysis_json=None):
        bpm = dsp.song_bpm_value(bpm, song_bpm, analysis_json)
        return (dsp.rhythmic_gate_stutter(audio, bpm, division, pattern, depth, smoothing_ms, mode, mix),)


class ComfyAudioDSPFoldClip(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFoldClip", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFoldClip", "Hybrid wavefolding and clipping distortion.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "fold_clip"
        return cls._finish({"drive_db": _float(section, "drive_db", 12.0, 0.0, 80.0, 0.1), "fold_amount": _float(section, "fold_amount", 2.0, 1.0, 16.0, 0.1), "clip_threshold": _float(section, "clip_threshold", 0.7, 0.01, 2.0, 0.01), "mode": _combo(section, "mode", ["blend", "fold_then_clip", "clip_then_fold"], "blend"), "output_gain_db": _float(section, "output_gain_db", -6.0, -60.0, 24.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, drive_db, fold_amount, clip_threshold, mode, output_gain_db, mix):
        return (dsp.fold_clip(audio, drive_db, fold_amount, clip_threshold, mode, output_gain_db, mix),)


class ComfyAudioDSPAmpSimulator(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAmpSimulator", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPAmpSimulator", "Simple amp and cabinet simulator with drive, tone, and presence.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "amp_simulator"
        return cls._finish({"drive_db": _float(section, "drive_db", 18.0, 0.0, 80.0, 0.1), "tone": _float(section, "tone", 0.55, 0.0, 1.0, 0.01), "cabinet": _combo(section, "cabinet", ["closed_back", "open_back"], "closed_back"), "presence": _float(section, "presence", 0.5, 0.0, 2.0, 0.01), "output_gain_db": _float(section, "output_gain_db", -9.0, -60.0, 24.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, drive_db, tone, cabinet, presence, output_gain_db, mix):
        return (dsp.amp_simulator(audio, drive_db, tone, cabinet, presence, output_gain_db, mix),)


class ComfyAudioDSPCrossoverDistortion(_AudioDSPNode):
    CATEGORY = CATEGORY_SATURATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPCrossoverDistortion", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPCrossoverDistortion", "Crossover distortion model with dead-zone threshold and slope.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "crossover_distortion"
        return cls._finish({"threshold": _float(section, "threshold", 0.08, 0.0, 0.5, 0.001), "slope": _float(section, "slope", 0.35, 0.0, 1.0, 0.01), "drive_db": _float(section, "drive_db", 6.0, -24.0, 60.0, 0.1), "output_gain_db": _float(section, "output_gain_db", -3.0, -60.0, 24.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, threshold, slope, drive_db, output_gain_db, mix):
        return (dsp.crossover_distortion(audio, threshold, slope, drive_db, output_gain_db, mix),)


class ComfyAudioDSPFormantShifter(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFormantShifter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFormantShifter", "Shifts spectral-envelope formants without intentionally changing timing.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "formant_shifter"
        return cls._finish({"shift_ratio": _float(section, "shift_ratio", 1.0, 0.25, 4.0, 0.01), "fft_size": _int(section, "fft_size", 2048, 128, 32768, 64), "hop_size": _int(section, "hop_size", 512, 16, 8192, 16), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, shift_ratio, fft_size, hop_size, mix):
        return (dsp.formant_shifter(audio, shift_ratio, fft_size, hop_size, mix),)


class ComfyAudioDSPPSOLAPitchShifter(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPSOLAPitchShifter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPSOLAPitchShifter", "Time-domain PSOLA-style pitch shifter option for monophonic material.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "psola_pitch_shifter"
        return cls._finish({"semitones": _float(section, "semitones", 0.0, -24.0, 24.0, 0.01), "cents": _float(section, "cents", 0.0, -100.0, 100.0, 0.1), "frame_ms": _float(section, "frame_ms", 30.0, 5.0, 80.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, semitones, cents, frame_ms, mix):
        return (dsp.psola_pitch_shifter(audio, semitones, cents, frame_ms, mix),)


class ComfyAudioDSPPolyphonicPitchCorrection(_AudioDSPNode):
    CATEGORY = CATEGORY_PITCH_TIME
    RETURN_NAMES = loc.return_names("ComfyAudioDSPPolyphonicPitchCorrection", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPPolyphonicPitchCorrection", "Spectral pitch-class correction for polyphonic material.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "polyphonic_pitch_correction"
        return cls._finish({"key": _combo(section, "key", PITCH_KEYS, "C"), "scale": _combo(section, "scale", PITCH_SCALES, "major"), "correction_amount": _float(section, "correction_amount", 0.5, 0.0, 1.0, 0.01), "attenuation_db": _float(section, "attenuation_db", 12.0, 0.0, 60.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section, {"song_key": _force_input(section, "song_key", "STRING"), "analysis_json": _force_input(section, "analysis_json", "STRING")})

    def process(self, audio, key, scale, correction_amount, attenuation_db, mix, song_key=None, analysis_json=None):
        if song_key not in (None, "") or analysis_json not in (None, ""):
            key, scale, _details = dsp.song_key_to_pitch_controls(song_key or "", analysis_json)
        return (dsp.polyphonic_pitch_correction(audio, key, scale, correction_amount, attenuation_db, mix),)


class ComfyAudioDSPTruePeakMeter(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPTruePeakMeter", ("audio", "true_peak_dbfs", "overload", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPTruePeakMeter", "Oversampled true-peak meter.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({"oversample": _int("true_peak_meter", "oversample", 4, 1, 16)}, "true_peak_meter")

    def process(self, audio, oversample):
        return dsp.true_peak_meter(audio, oversample)


class ComfyAudioDSPDynamicRangeDRMeter(_AudioDSPNode):
    CATEGORY = CATEGORY_METERING
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDynamicRangeDRMeter", ("audio", "dr", "peak_dbfs", "top_rms_dbfs", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPDynamicRangeDRMeter", "Crest-factor style dynamic range meter.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._finish({"block_ms": _float("dynamic_range_dr_meter", "block_ms", 3000.0, 50.0, 30000.0, 10.0)}, "dynamic_range_dr_meter")

    def process(self, audio, block_ms):
        return dsp.dynamic_range_dr_meter(audio, block_ms)


class ComfyAudioDSPVBAPPanner(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPVBAPPanner", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPVBAPPanner", "VBAP-style panner for arbitrary speaker angle layouts.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "vbap_panner"
        return cls._finish({"azimuth_deg": _float(section, "azimuth_deg", 0.0, -180.0, 180.0, 1.0), "speaker_angles_deg": _string(section, "speaker_angles_deg", "-30,30"), "spread": _float(section, "spread", 0.0, 0.0, 180.0, 1.0), "normalize": _bool(section, "normalize", True)}, section)

    def process(self, audio, azimuth_deg, speaker_angles_deg, spread, normalize):
        return (dsp.vbap_panner(audio, azimuth_deg, speaker_angles_deg, spread, normalize),)


class ComfyAudioDSPHigherOrderAmbisonicsEncoder(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHigherOrderAmbisonicsEncoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHigherOrderAmbisonicsEncoder", "Encodes mono audio to approximate higher-order Ambisonics channels.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "higher_order_ambisonics_encoder"
        return cls._finish({"order": _combo(section, "order", HOA_ORDERS, "2"), "azimuth_deg": _float(section, "azimuth_deg", 0.0, -180.0, 180.0, 1.0), "elevation_deg": _float(section, "elevation_deg", 0.0, -90.0, 90.0, 1.0), "gain_db": _float(section, "gain_db", 0.0, -24.0, 24.0, 0.1)}, section)

    def process(self, audio, order, azimuth_deg, elevation_deg, gain_db):
        return (dsp.higher_order_ambisonics_encoder(audio, order, azimuth_deg, elevation_deg, gain_db),)


class ComfyAudioDSPHigherOrderAmbisonicsDecoder(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHigherOrderAmbisonicsDecoder", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHigherOrderAmbisonicsDecoder", "Decodes approximate HOA channels to stereo or speaker-angle layouts.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "higher_order_ambisonics_decoder"
        return cls._finish({"order": _combo(section, "order", HOA_ORDERS, "2"), "mode": _combo(section, "mode", ["stereo", "speaker_layout"], "stereo"), "speaker_angles_deg": _string(section, "speaker_angles_deg", "-30,30"), "width": _float(section, "width", 1.0, 0.0, 4.0, 0.01)}, section)

    def process(self, audio, order, mode, speaker_angles_deg, width):
        return (dsp.higher_order_ambisonics_decoder(audio, order, mode, speaker_angles_deg, width),)


class ComfyAudioDSPHigherOrderAmbisonicsRotator(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPHigherOrderAmbisonicsRotator", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPHigherOrderAmbisonicsRotator", "Yaw-rotates approximate higher-order Ambisonics channels.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "higher_order_ambisonics_rotator"
        return cls._finish({"order": _combo(section, "order", HOA_ORDERS, "2"), "yaw_deg": _float(section, "yaw_deg", 0.0, -180.0, 180.0, 1.0)}, section)

    def process(self, audio, order, yaw_deg):
        return (dsp.higher_order_ambisonics_rotator(audio, order, yaw_deg),)


class ComfyAudioDSPSixDOFRenderer(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSixDOFRenderer", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSixDOFRenderer", "Simple 6DOF renderer using source/listener position and listener orientation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "six_dof_renderer"
        return cls._finish({"source_x": _float(section, "source_x", 0.0, -1000.0, 1000.0, 0.01), "source_y": _float(section, "source_y", 1.0, -1000.0, 1000.0, 0.01), "source_z": _float(section, "source_z", 0.0, -1000.0, 1000.0, 0.01), "listener_x": _float(section, "listener_x", 0.0, -1000.0, 1000.0, 0.01), "listener_y": _float(section, "listener_y", 0.0, -1000.0, 1000.0, 0.01), "listener_z": _float(section, "listener_z", 0.0, -1000.0, 1000.0, 0.01), "yaw_deg": _float(section, "yaw_deg", 0.0, -180.0, 180.0, 1.0), "pitch_deg": _float(section, "pitch_deg", 0.0, -90.0, 90.0, 1.0), "roll_deg": _float(section, "roll_deg", 0.0, -180.0, 180.0, 1.0), "room_mix": _float(section, "room_mix", 0.2, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, source_x, source_y, source_z, listener_x, listener_y, listener_z, yaw_deg, pitch_deg, roll_deg, room_mix):
        return (dsp.six_dof_renderer(audio, source_x, source_y, source_z, listener_x, listener_y, listener_z, yaw_deg, pitch_deg, roll_deg, room_mix),)


class ComfyAudioDSPBRIRConvolution(_AudioDSPNode):
    CATEGORY = CATEGORY_SPATIAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBRIRConvolution", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPBRIRConvolution", "Convolves mono audio with a stereo BRIR WAV.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "brir_convolution"
        return cls._finish({"brir_wav": _string(section, "brir_wav"), "normalize_ir": _bool(section, "normalize_ir", True), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, brir_wav, normalize_ir, mix):
        return (dsp.brir_convolution(audio, brir_wav, normalize_ir, mix),)


class ComfyAudioDSPFMOperator:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFMOperator", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPFMOperator", "FM operator tone generator.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "fm_operator"
        return {"required": {"carrier_hz": _float(section, "carrier_hz", 220.0, 0.01, 22000.0, 0.01), "modulator_hz": _float(section, "modulator_hz", 440.0, 0.01, 22000.0, 0.01), "modulation_index": _float(section, "modulation_index", 2.0, 0.0, 50.0, 0.01), "amplitude": _float(section, "amplitude", 0.25, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 1.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "channels": _int(section, "channels", 2, 1, 8)}}

    def process(self, carrier_hz, modulator_hz, modulation_index, amplitude, duration_s, sample_rate, channels):
        return (dsp.fm_operator(carrier_hz, modulator_hz, modulation_index, amplitude, duration_s, sample_rate, channels),)


class ComfyAudioDSPKarplusStrongString:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPKarplusStrongString", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPKarplusStrongString", "Plucked-string synthesis with Karplus-Strong feedback.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "karplus_strong_string"
        return {"required": {"frequency_hz": _float(section, "frequency_hz", 110.0, 1.0, 8000.0, 0.01), "decay": _float(section, "decay", 0.995, 0.0, 0.9999, 0.0001), "brightness": _float(section, "brightness", 0.7, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 2.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "seed": _int(section, "seed", 0, 0, 2147483647)}}

    def process(self, frequency_hz, decay, brightness, duration_s, sample_rate, seed):
        return (dsp.karplus_strong_string(frequency_hz, decay, brightness, duration_s, sample_rate, seed),)


class ComfyAudioDSPWavetableOscillator:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPWavetableOscillator", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPWavetableOscillator", "Wavetable oscillator with built-in table shapes.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "wavetable_oscillator"
        return {"required": {"wavetable": _combo(section, "wavetable", WAVETABLES, "sine"), "frequency_hz": _float(section, "frequency_hz", 440.0, 0.01, 22000.0, 0.01), "amplitude": _float(section, "amplitude", 0.25, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 1.0, 0.001, 3600.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 8000, 384000, 1), "table_size": _int(section, "table_size", 2048, 16, 65536, 16), "channels": _int(section, "channels", 2, 1, 8)}}

    def process(self, wavetable, frequency_hz, amplitude, duration_s, sample_rate, table_size, channels):
        return (dsp.wavetable_oscillator(wavetable, frequency_hz, amplitude, duration_s, sample_rate, table_size, channels),)


class ComfyAudioDSPSamplePlayer:
    CATEGORY = CATEGORY_GENERATORS
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSamplePlayer", ("audio",))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPSamplePlayer", "Loads a WAV sample, optionally trims, loops, gains, and resamples it.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "sample_player"
        return {"required": {"path": _string(section, "path"), "target_sample_rate": _int(section, "target_sample_rate", 44100, 0, 384000, 1), "start_s": _float(section, "start_s", 0.0, 0.0, 36000.0, 0.001), "duration_s": _float(section, "duration_s", 0.0, 0.0, 36000.0, 0.001), "gain_db": _float(section, "gain_db", 0.0, -60.0, 24.0, 0.1), "loop": _bool(section, "loop", False)}}

    def process(self, path, target_sample_rate, start_s, duration_s, gain_db, loop):
        return (dsp.sample_player(path, target_sample_rate, start_s, duration_s, gain_db, loop),)


class ComfyAudioDSPSpectralGate(_AudioDSPNode):
    CATEGORY = CATEGORY_SPECTRAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectralGate", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectralGate", "FFT-bin noise gate with threshold, reduction, smoothing, and mix.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectral_gate"
        return cls._finish({"threshold_db": _float(section, "threshold_db", -48.0, -120.0, 0.0, 0.5), "reduction_db": _float(section, "reduction_db", 36.0, 0.0, 120.0, 0.5), "fft_size": _int(section, "fft_size", 1024, 128, 8192, 128), "hop_size": _int(section, "hop_size", 256, 32, 4096, 32), "smoothing_bins": _int(section, "smoothing_bins", 3, 1, 64), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, threshold_db, reduction_db, fft_size, hop_size, smoothing_bins, mix):
        return (dsp.spectral_gate(audio, threshold_db, reduction_db, fft_size, hop_size, smoothing_bins, mix),)


class ComfyAudioDSPSpectralFreeze(_AudioDSPNode):
    CATEGORY = CATEGORY_SPECTRAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectralFreeze", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectralFreeze", "Freezes one FFT frame and resynthesizes it for a sustained spectral texture.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectral_freeze"
        return cls._finish({"freeze_time_s": _float(section, "freeze_time_s", 0.0, 0.0, 36000.0, 0.001), "duration_s": _float(section, "duration_s", 0.0, 0.0, 36000.0, 0.001), "fft_size": _int(section, "fft_size", 2048, 128, 16384, 128), "hop_size": _int(section, "hop_size", 512, 32, 8192, 32), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, freeze_time_s, duration_s, fft_size, hop_size, mix):
        return (dsp.spectral_freeze(audio, freeze_time_s, duration_s, fft_size, hop_size, mix),)


class ComfyAudioDSPFrequencyShifter(_AudioDSPNode):
    CATEGORY = CATEGORY_SPECTRAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPFrequencyShifter", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPFrequencyShifter", "Linear frequency shifter based on Hilbert analytic-signal modulation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "frequency_shifter"
        return cls._finish({"shift_hz": _float(section, "shift_hz", 25.0, -5000.0, 5000.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, shift_hz, mix):
        return (dsp.frequency_shifter(audio, shift_hz, mix),)


class ComfyAudioDSPSpectralBlur(_AudioDSPNode):
    CATEGORY = CATEGORY_SPECTRAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectralBlur", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectralBlur", "Blurs STFT magnitudes across frequency bins and time frames.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectral_blur"
        return cls._finish({"frequency_blur_bins": _int(section, "frequency_blur_bins", 9, 1, 128), "time_blur_frames": _int(section, "time_blur_frames", 5, 1, 128), "fft_size": _int(section, "fft_size", 1024, 128, 8192, 128), "hop_size": _int(section, "hop_size", 256, 32, 4096, 32), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, frequency_blur_bins, time_blur_frames, fft_size, hop_size, mix):
        return (dsp.spectral_blur(audio, frequency_blur_bins, time_blur_frames, fft_size, hop_size, mix),)


class ComfyAudioDSPSpectralNoiseReduction(_AudioDSPNode):
    CATEGORY = CATEGORY_SPECTRAL
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSpectralNoiseReduction", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPSpectralNoiseReduction", "FFT spectral subtraction using an initial noise-profile window.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "spectral_noise_reduction"
        return cls._finish({"noise_profile_s": _float(section, "noise_profile_s", 0.5, 0.001, 60.0, 0.001), "reduction_db": _float(section, "reduction_db", 18.0, 0.0, 80.0, 0.5), "sensitivity": _float(section, "sensitivity", 1.5, 0.1, 8.0, 0.1), "fft_size": _int(section, "fft_size", 1024, 128, 8192, 128), "hop_size": _int(section, "hop_size", 256, 32, 4096, 32), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, noise_profile_s, reduction_db, sensitivity, fft_size, hop_size, mix):
        return (dsp.spectral_noise_reduction(audio, noise_profile_s, reduction_db, sensitivity, fft_size, hop_size, mix),)


class ComfyAudioDSPLFOSource:
    CATEGORY = CATEGORY_MOD_SOURCES
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPLFOSource", ("control_audio", "current_value", "values"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPLFOSource", "Standalone LFO control source exported as audio-rate control plus sampled values.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "lfo_source"
        return {"required": {"rate_hz": _float(section, "rate_hz", 1.0, 0.001, 200.0, 0.001), "depth": _float(section, "depth", 1.0, 0.0, 10.0, 0.01), "offset": _float(section, "offset", 0.0, -10.0, 10.0, 0.01), "waveform_shape": _combo(section, "waveform_shape", MOD_SOURCE_WAVEFORMS, "sine"), "duration_s": _float(section, "duration_s", 5.0, 0.001, 36000.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 1000, 384000, 1), "points": _int(section, "points", 128, 4, 4096, 1)}}

    def process(self, rate_hz, depth, offset, waveform_shape, duration_s, sample_rate, points):
        return dsp.lfo_source(rate_hz, depth, offset, waveform_shape, duration_s, sample_rate, points)


class ComfyAudioDSPADSREnvelopeGenerator:
    CATEGORY = CATEGORY_MOD_SOURCES
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPADSREnvelopeGenerator", ("control_audio", "current_value", "values"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPADSREnvelopeGenerator", "ADSR envelope generator for audio-rate parameter automation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "adsr_envelope_generator"
        return {"required": {"attack_ms": _float(section, "attack_ms", 10.0, 0.0, 60000.0, 0.1), "decay_ms": _float(section, "decay_ms", 120.0, 0.0, 60000.0, 0.1), "sustain_level": _float(section, "sustain_level", 0.7, 0.0, 1.0, 0.01), "release_ms": _float(section, "release_ms", 300.0, 0.0, 60000.0, 0.1), "gate_s": _float(section, "gate_s", 1.0, 0.0, 36000.0, 0.001), "duration_s": _float(section, "duration_s", 2.0, 0.001, 36000.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 1000, 384000, 1), "points": _int(section, "points", 128, 4, 4096, 1)}}

    def process(self, attack_ms, decay_ms, sustain_level, release_ms, gate_s, duration_s, sample_rate, points):
        return dsp.adsr_envelope_generator(attack_ms, decay_ms, sustain_level, release_ms, gate_s, duration_s, sample_rate, points)


class ComfyAudioDSPSampleAndHold:
    CATEGORY = CATEGORY_MOD_SOURCES
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSampleAndHold", ("control_audio", "current_value", "values"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPSampleAndHold", "Random stepped modulation source with optional smoothing.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "sample_and_hold"
        return {"required": {"rate_hz": _float(section, "rate_hz", 4.0, 0.001, 200.0, 0.001), "smoothing": _float(section, "smoothing", 0.0, 0.0, 1.0, 0.01), "seed": _int(section, "seed", 0, 0, 2147483647, 1), "duration_s": _float(section, "duration_s", 5.0, 0.001, 36000.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 1000, 384000, 1), "points": _int(section, "points", 128, 4, 4096, 1)}}

    def process(self, rate_hz, smoothing, seed, duration_s, sample_rate, points):
        return dsp.sample_and_hold(rate_hz, smoothing, seed, duration_s, sample_rate, points)


class ComfyAudioDSPStepSequencer:
    CATEGORY = CATEGORY_MOD_SOURCES
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPStepSequencer", ("control_audio", "current_value", "values"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPStepSequencer", "Comma-separated step sequencer for shared parameter modulation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "step_sequencer"
        return {"required": {"sequence": _string(section, "sequence", "0, 0.5, 1, 0.5", multiline=True), "bpm": _float(section, "bpm", 120.0, 1.0, 400.0, 0.1), "step_value": _combo(section, "step_value", ["1/1", "1/2", "1/4", "1/8", "1/16"], "1/16"), "glide": _float(section, "glide", 0.0, 0.0, 1.0, 0.01), "duration_s": _float(section, "duration_s", 5.0, 0.001, 36000.0, 0.001), "sample_rate": _int(section, "sample_rate", 44100, 1000, 384000, 1), "points": _int(section, "points", 128, 4, 4096, 1)}, "optional": {"song_bpm": _force_input(section, "song_bpm", "FLOAT"), "analysis_json": _force_input(section, "analysis_json", "STRING")}}

    def process(self, sequence, bpm, step_value, glide, duration_s, sample_rate, points, song_bpm=None, analysis_json=None):
        bpm = dsp.song_bpm_value(bpm, song_bpm, analysis_json)
        return dsp.step_sequencer(sequence, bpm, step_value, glide, duration_s, sample_rate, points)


class ComfyAudioDSPDeClip(_AudioDSPNode):
    CATEGORY = CATEGORY_RESTORATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDeClip", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDeClip", "Repairs short clipped regions with nonlinear masking and interpolation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "de_clip"
        return cls._finish({"threshold": _float(section, "threshold", 0.98, 0.05, 0.999, 0.001), "repair_ms": _float(section, "repair_ms", 0.5, 0.0, 20.0, 0.01), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, threshold, repair_ms, mix):
        return (dsp.de_clip(audio, threshold, repair_ms, mix),)


class ComfyAudioDSPDeReverb(_AudioDSPNode):
    CATEGORY = CATEGORY_RESTORATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDeReverb", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDeReverb", "Reduces diffuse reverb tails using STFT transient-preserving attenuation.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "de_reverb"
        return cls._finish({"strength": _float(section, "strength", 0.5, 0.0, 1.0, 0.01), "fft_size": _int(section, "fft_size", 1024, 128, 8192, 128), "hop_size": _int(section, "hop_size", 256, 32, 4096, 32), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, strength, fft_size, hop_size, mix):
        return (dsp.de_reverb(audio, strength, fft_size, hop_size, mix),)


class ComfyAudioDSPBandwidthExtension(_AudioDSPNode):
    CATEGORY = CATEGORY_RESTORATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBandwidthExtension", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPBandwidthExtension", "Adds synthesized high-frequency harmonics above a crossover.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "bandwidth_extension"
        return cls._finish({"crossover_hz": _float(section, "crossover_hz", 6000.0, 1000.0, 40000.0, 1.0), "amount": _float(section, "amount", 0.5, 0.0, 1.0, 0.01), "drive_db": _float(section, "drive_db", 12.0, 0.0, 60.0, 0.1), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, crossover_hz, amount, drive_db, mix):
        return (dsp.bandwidth_extension(audio, crossover_hz, amount, drive_db, mix),)


class ComfyAudioDSPDenoiser(_AudioDSPNode):
    CATEGORY = CATEGORY_RESTORATION
    RETURN_NAMES = loc.return_names("ComfyAudioDSPDenoiser", ("audio",))
    DESCRIPTION = loc.description("ComfyAudioDSPDenoiser", "Broadband background-noise reduction using spectral subtraction or Wiener-style gain.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "denoiser"
        return cls._finish({"method": _combo(section, "method", DENOISER_METHODS, "wiener"), "noise_profile_s": _float(section, "noise_profile_s", 0.5, 0.001, 60.0, 0.001), "strength": _float(section, "strength", 0.6, 0.0, 1.0, 0.01), "fft_size": _int(section, "fft_size", 1024, 128, 8192, 128), "hop_size": _int(section, "hop_size", 256, 32, 4096, 32), "mix": _float(section, "mix", 1.0, 0.0, 1.0, 0.01)}, section)

    def process(self, audio, method, noise_profile_s, strength, fft_size, hop_size, mix):
        return (dsp.denoiser(audio, method, noise_profile_s, strength, fft_size, hop_size, mix),)


class ComfyAudioDSPAudioFeatureToText(_AudioDSPNode):
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioFeatureToText", ("audio", "text"))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioFeatureToText", "Extracts BPM, key, loudness, and timbre features as prompt-ready text.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "audio_feature_to_text"
        return cls._finish({"include_bpm": _bool(section, "include_bpm", True), "include_key": _bool(section, "include_key", True), "include_loudness": _bool(section, "include_loudness", True), "include_timbre": _bool(section, "include_timbre", True)}, section)

    def process(self, audio, include_bpm, include_key, include_loudness, include_timbre):
        return dsp.audio_feature_to_text(audio, include_bpm, include_key, include_loudness, include_timbre)


class ComfyAudioDSPBeatSlicer(_AudioDSPNode):
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPBeatSlicer", ("audio", "slices", "slice_count"))
    DESCRIPTION = loc.description("ComfyAudioDSPBeatSlicer", "Detects transient slice ranges for workflow scheduling.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "beat_slicer"
        return cls._finish({"sensitivity": _float(section, "sensitivity", 0.7, 0.0, 1.0, 0.01), "min_gap_ms": _float(section, "min_gap_ms", 80.0, 1.0, 5000.0, 1.0), "padding_ms": _float(section, "padding_ms", 0.0, 0.0, 5000.0, 1.0)}, section)

    def process(self, audio, sensitivity, min_gap_ms, padding_ms):
        return dsp.beat_slicer(audio, sensitivity, min_gap_ms, padding_ms)


class ComfyAudioDSPAudioQualityEstimator(_AudioDSPNode):
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPAudioQualityEstimator", ("audio", "score", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPAudioQualityEstimator", "Estimates audio quality with no-reference heuristics or an optional reference signal.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "audio_quality_estimator"
        return {"required": {"audio": _audio_input(section), "quality_mode": _combo(section, "quality_mode", ["music", "speech", "general"], "general")}, "optional": {"reference_audio": _optional_audio(section, "reference_audio")}}

    def process(self, audio, quality_mode, reference_audio=None):
        return dsp.audio_quality_estimator(audio, quality_mode, reference_audio)


class ComfyAudioDSPSongAnalysisToDSPControls:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "STRING", "SA_FLOAT_LIST", "SA_INT_LIST", "SA_SEGMENTS", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSongAnalysisToDSPControls", ("bpm", "key_text", "key", "scale", "beat_times", "downbeats", "segments", "details"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPSongAnalysisToDSPControls", "Parses Song-Analyst analysis_json into reusable DSP control outputs.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "song_analysis_to_dsp_controls"
        return {"required": {"analysis_json": _force_input(section, "analysis_json", "STRING")}}

    def process(self, analysis_json):
        return dsp.analysis_to_dsp_controls(analysis_json)


class ComfyAudioDSPSongKeyToPitchControls:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSongKeyToPitchControls", ("key", "scale", "details"))
    FUNCTION = "process"
    DESCRIPTION = loc.description("ComfyAudioDSPSongKeyToPitchControls", "Converts Song-Analyst key text into Audio-DSP pitch correction controls.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "song_key_to_pitch_controls"
        return {"required": {"key_text": _string(section, "key_text", "C major")}, "optional": {"analysis_json": _force_input(section, "analysis_json", "STRING")}}

    def process(self, key_text, analysis_json=None):
        return dsp.song_key_to_pitch_controls(key_text, analysis_json)


class ComfyAudioDSPSongSegmentSelector(_AudioDSPNode):
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSongSegmentSelector", ("audio", "start_s", "end_s", "details"))
    DESCRIPTION = loc.description("ComfyAudioDSPSongSegmentSelector", "Selects and crops a Song-Analyst structure or similarity segment.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "song_segment_selector"
        return cls._finish({
            "segment_index": _int(section, "segment_index", 0, 0, 9999),
            "label_filter": _string(section, "label_filter", ""),
            "padding_s": _float(section, "padding_s", 0.0, 0.0, 3600.0, 0.001),
        }, section, {
            "song_segments": _force_input(section, "song_segments", "SA_SEGMENTS"),
            "analysis_json": _force_input(section, "analysis_json", "STRING"),
        })

    def process(self, audio, segment_index, label_filter, padding_s, song_segments=None, analysis_json=None):
        return dsp.song_segment_selector(audio, segment_index, label_filter, padding_s, song_segments, analysis_json)


class ComfyAudioDSPSongBeatGridSlicer(_AudioDSPNode):
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = loc.return_names("ComfyAudioDSPSongBeatGridSlicer", ("audio", "slices", "slice_count"))
    DESCRIPTION = loc.description("ComfyAudioDSPSongBeatGridSlicer", "Builds slice ranges from Song-Analyst beat times and optional downbeat flags.")

    @classmethod
    def INPUT_TYPES(cls):
        section = "song_beat_grid_slicer"
        return cls._finish({
            "beats_per_slice": _int(section, "beats_per_slice", 4, 1, 64),
            "padding_ms": _float(section, "padding_ms", 0.0, 0.0, 60000.0, 0.1),
            "prefer_downbeats": _bool(section, "prefer_downbeats", True),
        }, section, {
            "beat_times": _force_input(section, "beat_times", "SA_FLOAT_LIST"),
            "downbeats": _force_input(section, "downbeats", "SA_INT_LIST"),
            "analysis_json": _force_input(section, "analysis_json", "STRING"),
        })

    def process(self, audio, beats_per_slice, padding_ms, prefer_downbeats, beat_times=None, downbeats=None, analysis_json=None):
        return dsp.song_beat_grid_slicer(audio, beats_per_slice, padding_ms, prefer_downbeats, beat_times, downbeats, analysis_json)


NODE_CLASS_MAPPINGS = {
    "ComfyAudioDSPCompressor": ComfyAudioDSPCompressor,
    "ComfyAudioDSPLimiter": ComfyAudioDSPLimiter,
    "ComfyAudioDSPMidSideCompressor": ComfyAudioDSPMidSideCompressor,
    "ComfyAudioDSPNoiseGate": ComfyAudioDSPNoiseGate,
    "ComfyAudioDSPExpander": ComfyAudioDSPExpander,
    "ComfyAudioDSPTransientShaper": ComfyAudioDSPTransientShaper,
    "ComfyAudioDSPDeEsser": ComfyAudioDSPDeEsser,
    "ComfyAudioDSPMultiBandCompressor": ComfyAudioDSPMultiBandCompressor,
    "ComfyAudioDSPMultiBandLimiter": ComfyAudioDSPMultiBandLimiter,
    "ComfyAudioDSPAutoGainLeveler": ComfyAudioDSPAutoGainLeveler,
    "ComfyAudioDSPLoudnessNormalizer": ComfyAudioDSPLoudnessNormalizer,
    "ComfyAudioDSPLowHighShelf": ComfyAudioDSPLowHighShelf,
    "ComfyAudioDSPPeakBellFilter": ComfyAudioDSPPeakBellFilter,
    "ComfyAudioDSPLowHighPassFilter": ComfyAudioDSPLowHighPassFilter,
    "ComfyAudioDSPBandPassStopFilter": ComfyAudioDSPBandPassStopFilter,
    "ComfyAudioDSPNotchFilter": ComfyAudioDSPNotchFilter,
    "ComfyAudioDSPThreeBandEQ": ComfyAudioDSPThreeBandEQ,
    "ComfyAudioDSPParametricEQ": ComfyAudioDSPParametricEQ,
    "ComfyAudioDSPGraphicEQ": ComfyAudioDSPGraphicEQ,
    "ComfyAudioDSPTiltEQ": ComfyAudioDSPTiltEQ,
    "ComfyAudioDSPRiaaEQ": ComfyAudioDSPRiaaEQ,
    "ComfyAudioDSPResonantPassFilter": ComfyAudioDSPResonantPassFilter,
    "ComfyAudioDSPMatchEQ": ComfyAudioDSPMatchEQ,
    "ComfyAudioDSPConvolutionReverb": ComfyAudioDSPConvolutionReverb,
    "ComfyAudioDSPIRManager": ComfyAudioDSPIRManager,
    "ComfyAudioDSPSchroederReverb": ComfyAudioDSPSchroederReverb,
    "ComfyAudioDSPFreeverbMoorerReverb": ComfyAudioDSPFreeverbMoorerReverb,
    "ComfyAudioDSPSpringReverb": ComfyAudioDSPSpringReverb,
    "ComfyAudioDSPPlateReverb": ComfyAudioDSPPlateReverb,
    "ComfyAudioDSPGatedReverb": ComfyAudioDSPGatedReverb,
    "ComfyAudioDSPReverseReverb": ComfyAudioDSPReverseReverb,
    "ComfyAudioDSPFDNReverb": ComfyAudioDSPFDNReverb,
    "ComfyAudioDSPSimpleDelay": ComfyAudioDSPSimpleDelay,
    "ComfyAudioDSPTempoSyncedDelay": ComfyAudioDSPTempoSyncedDelay,
    "ComfyAudioDSPPingPongDelay": ComfyAudioDSPPingPongDelay,
    "ComfyAudioDSPMultiTapDelay": ComfyAudioDSPMultiTapDelay,
    "ComfyAudioDSPDubDelay": ComfyAudioDSPDubDelay,
    "ComfyAudioDSPFilteredDelay": ComfyAudioDSPFilteredDelay,
    "ComfyAudioDSPStereoSpreadDelay": ComfyAudioDSPStereoSpreadDelay,
    "ComfyAudioDSPEchoplexTapeEcho": ComfyAudioDSPEchoplexTapeEcho,
    "ComfyAudioDSPChorus": ComfyAudioDSPChorus,
    "ComfyAudioDSPFlanger": ComfyAudioDSPFlanger,
    "ComfyAudioDSPPhaser": ComfyAudioDSPPhaser,
    "ComfyAudioDSPTremolo": ComfyAudioDSPTremolo,
    "ComfyAudioDSPVibrato": ComfyAudioDSPVibrato,
    "ComfyAudioDSPRotarySpeaker": ComfyAudioDSPRotarySpeaker,
    "ComfyAudioDSPRingModulator": ComfyAudioDSPRingModulator,
    "ComfyAudioDSPAutoPanner": ComfyAudioDSPAutoPanner,
    "ComfyAudioDSPUniVibe": ComfyAudioDSPUniVibe,
    "ComfyAudioDSPSoftClipper": ComfyAudioDSPSoftClipper,
    "ComfyAudioDSPHardClipper": ComfyAudioDSPHardClipper,
    "ComfyAudioDSPTubeSaturation": ComfyAudioDSPTubeSaturation,
    "ComfyAudioDSPTapeSaturation": ComfyAudioDSPTapeSaturation,
    "ComfyAudioDSPFuzz": ComfyAudioDSPFuzz,
    "ComfyAudioDSPBitCrusher": ComfyAudioDSPBitCrusher,
    "ComfyAudioDSPOverdriveDistortion": ComfyAudioDSPOverdriveDistortion,
    "ComfyAudioDSPWavefolder": ComfyAudioDSPWavefolder,
    "ComfyAudioDSPExciterEnhancer": ComfyAudioDSPExciterEnhancer,
    "ComfyAudioDSPPitchShifter": ComfyAudioDSPPitchShifter,
    "ComfyAudioDSPTimeStretcher": ComfyAudioDSPTimeStretcher,
    "ComfyAudioDSPResamplerClassic": ComfyAudioDSPResamplerClassic,
    "ComfyAudioDSPHarmonizer": ComfyAudioDSPHarmonizer,
    "ComfyAudioDSPPitchCorrection": ComfyAudioDSPPitchCorrection,
    "ComfyAudioDSPVarispeedPlayer": ComfyAudioDSPVarispeedPlayer,
    "ComfyAudioDSPPannerBalance": ComfyAudioDSPPannerBalance,
    "ComfyAudioDSPStereoWidth": ComfyAudioDSPStereoWidth,
    "ComfyAudioDSPMidSideEncoder": ComfyAudioDSPMidSideEncoder,
    "ComfyAudioDSPMidSideDecoder": ComfyAudioDSPMidSideDecoder,
    "ComfyAudioDSPMidSideEQ": ComfyAudioDSPMidSideEQ,
    "ComfyAudioDSPStereoEnhancerHaas": ComfyAudioDSPStereoEnhancerHaas,
    "ComfyAudioDSPSwapChannels": ComfyAudioDSPSwapChannels,
    "ComfyAudioDSPMonoMaker": ComfyAudioDSPMonoMaker,
    "ComfyAudioDSPBinauralPanner": ComfyAudioDSPBinauralPanner,
    "ComfyAudioDSPHRTFConvolution": ComfyAudioDSPHRTFConvolution,
    "ComfyAudioDSPAmbisonicsEncoder": ComfyAudioDSPAmbisonicsEncoder,
    "ComfyAudioDSPAmbisonicsDecoder": ComfyAudioDSPAmbisonicsDecoder,
    "ComfyAudioDSPAmbisonicsRotator": ComfyAudioDSPAmbisonicsRotator,
    "ComfyAudioDSPDistanceSimulator": ComfyAudioDSPDistanceSimulator,
    "ComfyAudioDSPDopplerEffect": ComfyAudioDSPDopplerEffect,
    "ComfyAudioDSPRMSMeter": ComfyAudioDSPRMSMeter,
    "ComfyAudioDSPPeakMeter": ComfyAudioDSPPeakMeter,
    "ComfyAudioDSPLUFSMeter": ComfyAudioDSPLUFSMeter,
    "ComfyAudioDSPLoudnessGraph": ComfyAudioDSPLoudnessGraph,
    "ComfyAudioDSPSpectralAnalyzer": ComfyAudioDSPSpectralAnalyzer,
    "ComfyAudioDSPSpectrogramVisualizer": ComfyAudioDSPSpectrogramVisualizer,
    "ComfyAudioDSPWaveformVisualizer": ComfyAudioDSPWaveformVisualizer,
    "ComfyAudioDSPPhaseCorrelationMeter": ComfyAudioDSPPhaseCorrelationMeter,
    "ComfyAudioDSPGoniometerVectorscope": ComfyAudioDSPGoniometerVectorscope,
    "ComfyAudioDSPBPMTempoDetector": ComfyAudioDSPBPMTempoDetector,
    "ComfyAudioDSPKeyPitchDetector": ComfyAudioDSPKeyPitchDetector,
    "ComfyAudioDSPTransientOnsetDetector": ComfyAudioDSPTransientOnsetDetector,
    "ComfyAudioDSPSilenceDetector": ComfyAudioDSPSilenceDetector,
    "ComfyAudioDSPSineWaveGenerator": ComfyAudioDSPSineWaveGenerator,
    "ComfyAudioDSPNoiseGenerator": ComfyAudioDSPNoiseGenerator,
    "ComfyAudioDSPSweepChirp": ComfyAudioDSPSweepChirp,
    "ComfyAudioDSPImpulse": ComfyAudioDSPImpulse,
    "ComfyAudioDSPOscillatorMultiWave": ComfyAudioDSPOscillatorMultiWave,
    "ComfyAudioDSPClickTrackMetronome": ComfyAudioDSPClickTrackMetronome,
    "ComfyAudioDSPAudioMixer": ComfyAudioDSPAudioMixer,
    "ComfyAudioDSPAudioRouterSelector": ComfyAudioDSPAudioRouterSelector,
    "ComfyAudioDSPAudioSplitter": ComfyAudioDSPAudioSplitter,
    "ComfyAudioDSPAudioMerger": ComfyAudioDSPAudioMerger,
    "ComfyAudioDSPCrossfader": ComfyAudioDSPCrossfader,
    "ComfyAudioDSPSidechainGateCompressor": ComfyAudioDSPSidechainGateCompressor,
    "ComfyAudioDSPSendReturnLoop": ComfyAudioDSPSendReturnLoop,
    "ComfyAudioDSPParallelProcessingRouter": ComfyAudioDSPParallelProcessingRouter,
    "ComfyAudioDSPParallelReturnMixer": ComfyAudioDSPParallelReturnMixer,
    "ComfyAudioDSPGainTrim": ComfyAudioDSPGainTrim,
    "ComfyAudioDSPPhaseInverter": ComfyAudioDSPPhaseInverter,
    "ComfyAudioDSPDCOffsetRemover": ComfyAudioDSPDCOffsetRemover,
    "ComfyAudioDSPFadeInOut": ComfyAudioDSPFadeInOut,
    "ComfyAudioDSPAudioTrimCrop": ComfyAudioDSPAudioTrimCrop,
    "ComfyAudioDSPSilenceTrimmer": ComfyAudioDSPSilenceTrimmer,
    "ComfyAudioDSPNormalize": ComfyAudioDSPNormalize,
    "ComfyAudioDSPResampleChangeSampleRate": ComfyAudioDSPResampleChangeSampleRate,
    "ComfyAudioDSPFormatConverter": ComfyAudioDSPFormatConverter,
    "ComfyAudioDSPAudioInfo": ComfyAudioDSPAudioInfo,
    "ComfyAudioDSPDelayCompensation": ComfyAudioDSPDelayCompensation,
    "ComfyAudioDSPLoopDuplicator": ComfyAudioDSPLoopDuplicator,
    "ComfyAudioDSPReverseAudio": ComfyAudioDSPReverseAudio,
    "ComfyAudioDSPDither": ComfyAudioDSPDither,
    "ComfyAudioDSPDynamicEQ": ComfyAudioDSPDynamicEQ,
    "ComfyAudioDSPVocoder": ComfyAudioDSPVocoder,
    "ComfyAudioDSPEnvelopeFollowerOutput": ComfyAudioDSPEnvelopeFollowerOutput,
    "ComfyAudioDSPMultibandCrossover": ComfyAudioDSPMultibandCrossover,
    "ComfyAudioDSPDeclickDecrackle": ComfyAudioDSPDeclickDecrackle,
    "ComfyAudioDSPSpectralSmoothingContrast": ComfyAudioDSPSpectralSmoothingContrast,
    "ComfyAudioDSPHumRemover": ComfyAudioDSPHumRemover,
    "ComfyAudioDSPPhaseRotatorAllpass": ComfyAudioDSPPhaseRotatorAllpass,
    "ComfyAudioDSPGranularProcessor": ComfyAudioDSPGranularProcessor,
    "ComfyAudioDSPMathSignalMixer": ComfyAudioDSPMathSignalMixer,
    "ComfyAudioDSPUpwardCompressor": ComfyAudioDSPUpwardCompressor,
    "ComfyAudioDSPParallelCompressionMix": ComfyAudioDSPParallelCompressionMix,
    "ComfyAudioDSPLinearPhaseEQ": ComfyAudioDSPLinearPhaseEQ,
    "ComfyAudioDSPCombFilter": ComfyAudioDSPCombFilter,
    "ComfyAudioDSPShimmerReverb": ComfyAudioDSPShimmerReverb,
    "ComfyAudioDSPReverseDelay": ComfyAudioDSPReverseDelay,
    "ComfyAudioDSPGranularDelay": ComfyAudioDSPGranularDelay,
    "ComfyAudioDSPSlapEcho": ComfyAudioDSPSlapEcho,
    "ComfyAudioDSPBarberpoleFlanger": ComfyAudioDSPBarberpoleFlanger,
    "ComfyAudioDSPAutoFilter": ComfyAudioDSPAutoFilter,
    "ComfyAudioDSPAutoWah": ComfyAudioDSPAutoWah,
    "ComfyAudioDSPRhythmicGateStutter": ComfyAudioDSPRhythmicGateStutter,
    "ComfyAudioDSPFoldClip": ComfyAudioDSPFoldClip,
    "ComfyAudioDSPAmpSimulator": ComfyAudioDSPAmpSimulator,
    "ComfyAudioDSPCrossoverDistortion": ComfyAudioDSPCrossoverDistortion,
    "ComfyAudioDSPFormantShifter": ComfyAudioDSPFormantShifter,
    "ComfyAudioDSPPSOLAPitchShifter": ComfyAudioDSPPSOLAPitchShifter,
    "ComfyAudioDSPPolyphonicPitchCorrection": ComfyAudioDSPPolyphonicPitchCorrection,
    "ComfyAudioDSPTruePeakMeter": ComfyAudioDSPTruePeakMeter,
    "ComfyAudioDSPDynamicRangeDRMeter": ComfyAudioDSPDynamicRangeDRMeter,
    "ComfyAudioDSPVBAPPanner": ComfyAudioDSPVBAPPanner,
    "ComfyAudioDSPHigherOrderAmbisonicsEncoder": ComfyAudioDSPHigherOrderAmbisonicsEncoder,
    "ComfyAudioDSPHigherOrderAmbisonicsDecoder": ComfyAudioDSPHigherOrderAmbisonicsDecoder,
    "ComfyAudioDSPHigherOrderAmbisonicsRotator": ComfyAudioDSPHigherOrderAmbisonicsRotator,
    "ComfyAudioDSPSixDOFRenderer": ComfyAudioDSPSixDOFRenderer,
    "ComfyAudioDSPBRIRConvolution": ComfyAudioDSPBRIRConvolution,
    "ComfyAudioDSPFMOperator": ComfyAudioDSPFMOperator,
    "ComfyAudioDSPKarplusStrongString": ComfyAudioDSPKarplusStrongString,
    "ComfyAudioDSPWavetableOscillator": ComfyAudioDSPWavetableOscillator,
    "ComfyAudioDSPSamplePlayer": ComfyAudioDSPSamplePlayer,
    "ComfyAudioDSPSpectralGate": ComfyAudioDSPSpectralGate,
    "ComfyAudioDSPSpectralFreeze": ComfyAudioDSPSpectralFreeze,
    "ComfyAudioDSPFrequencyShifter": ComfyAudioDSPFrequencyShifter,
    "ComfyAudioDSPSpectralBlur": ComfyAudioDSPSpectralBlur,
    "ComfyAudioDSPSpectralNoiseReduction": ComfyAudioDSPSpectralNoiseReduction,
    "ComfyAudioDSPLFOSource": ComfyAudioDSPLFOSource,
    "ComfyAudioDSPADSREnvelopeGenerator": ComfyAudioDSPADSREnvelopeGenerator,
    "ComfyAudioDSPSampleAndHold": ComfyAudioDSPSampleAndHold,
    "ComfyAudioDSPStepSequencer": ComfyAudioDSPStepSequencer,
    "ComfyAudioDSPDeClip": ComfyAudioDSPDeClip,
    "ComfyAudioDSPDeReverb": ComfyAudioDSPDeReverb,
    "ComfyAudioDSPBandwidthExtension": ComfyAudioDSPBandwidthExtension,
    "ComfyAudioDSPDenoiser": ComfyAudioDSPDenoiser,
    "ComfyAudioDSPAudioFeatureToText": ComfyAudioDSPAudioFeatureToText,
    "ComfyAudioDSPBeatSlicer": ComfyAudioDSPBeatSlicer,
    "ComfyAudioDSPAudioQualityEstimator": ComfyAudioDSPAudioQualityEstimator,
    "ComfyAudioDSPSongAnalysisToDSPControls": ComfyAudioDSPSongAnalysisToDSPControls,
    "ComfyAudioDSPSongKeyToPitchControls": ComfyAudioDSPSongKeyToPitchControls,
    "ComfyAudioDSPSongSegmentSelector": ComfyAudioDSPSongSegmentSelector,
    "ComfyAudioDSPSongBeatGridSlicer": ComfyAudioDSPSongBeatGridSlicer,
}

_DISPLAY_FALLBACKS = {
    "ComfyAudioDSPCompressor": "Compressor",
    "ComfyAudioDSPLimiter": "Limiter",
    "ComfyAudioDSPMidSideCompressor": "M/S Compressor",
    "ComfyAudioDSPNoiseGate": "Noise Gate",
    "ComfyAudioDSPExpander": "Expander",
    "ComfyAudioDSPTransientShaper": "Transient Shaper",
    "ComfyAudioDSPDeEsser": "De-Esser",
    "ComfyAudioDSPMultiBandCompressor": "Multi-band Compressor",
    "ComfyAudioDSPMultiBandLimiter": "Multi-band Limiter",
    "ComfyAudioDSPAutoGainLeveler": "Auto Gain / Leveler",
    "ComfyAudioDSPLoudnessNormalizer": "Loudness Normalizer (LUFS)",
    "ComfyAudioDSPLowHighShelf": "Low Shelf / High Shelf",
    "ComfyAudioDSPPeakBellFilter": "Peak / Bell Filter",
    "ComfyAudioDSPLowHighPassFilter": "Low Pass / High Pass Filter",
    "ComfyAudioDSPBandPassStopFilter": "Band Pass / Band Stop Filter",
    "ComfyAudioDSPNotchFilter": "Notch Filter",
    "ComfyAudioDSPThreeBandEQ": "3-Band EQ",
    "ComfyAudioDSPParametricEQ": "Parametric EQ (N bands)",
    "ComfyAudioDSPGraphicEQ": "Graphic EQ (10/15/31 bands)",
    "ComfyAudioDSPTiltEQ": "Tilt EQ",
    "ComfyAudioDSPRiaaEQ": "RIAA EQ",
    "ComfyAudioDSPResonantPassFilter": "High Pass / Low Pass with Resonance",
    "ComfyAudioDSPMatchEQ": "Match EQ",
    "ComfyAudioDSPConvolutionReverb": "Convolution Reverb",
    "ComfyAudioDSPIRManager": "IR Manager",
    "ComfyAudioDSPSchroederReverb": "Schroeder Reverb",
    "ComfyAudioDSPFreeverbMoorerReverb": "Freeverb / Moorer Reverb",
    "ComfyAudioDSPSpringReverb": "Spring Reverb Sim",
    "ComfyAudioDSPPlateReverb": "Plate Reverb Sim",
    "ComfyAudioDSPGatedReverb": "Gated Reverb",
    "ComfyAudioDSPReverseReverb": "Reverse Reverb",
    "ComfyAudioDSPFDNReverb": "FDN Reverb",
    "ComfyAudioDSPSimpleDelay": "Simple Delay",
    "ComfyAudioDSPTempoSyncedDelay": "Tempo-synced Delay",
    "ComfyAudioDSPPingPongDelay": "Ping-Pong Delay",
    "ComfyAudioDSPMultiTapDelay": "Multi-tap Delay",
    "ComfyAudioDSPDubDelay": "Dub Delay",
    "ComfyAudioDSPFilteredDelay": "Filtered Delay",
    "ComfyAudioDSPStereoSpreadDelay": "Stereo Spread Delay",
    "ComfyAudioDSPEchoplexTapeEcho": "Echoplex Tape Echo",
    "ComfyAudioDSPChorus": "Chorus",
    "ComfyAudioDSPFlanger": "Flanger",
    "ComfyAudioDSPPhaser": "Phaser",
    "ComfyAudioDSPTremolo": "Tremolo",
    "ComfyAudioDSPVibrato": "Vibrato",
    "ComfyAudioDSPRotarySpeaker": "Rotary Speaker (Leslie)",
    "ComfyAudioDSPRingModulator": "Ring Modulator",
    "ComfyAudioDSPAutoPanner": "Auto Panner",
    "ComfyAudioDSPUniVibe": "Uni-vibe",
    "ComfyAudioDSPSoftClipper": "Soft Clipper",
    "ComfyAudioDSPHardClipper": "Hard Clipper",
    "ComfyAudioDSPTubeSaturation": "Tube Saturation",
    "ComfyAudioDSPTapeSaturation": "Tape Saturation",
    "ComfyAudioDSPFuzz": "Fuzz",
    "ComfyAudioDSPBitCrusher": "Bit Crusher",
    "ComfyAudioDSPOverdriveDistortion": "Overdrive / Distortion",
    "ComfyAudioDSPWavefolder": "Wavefolder",
    "ComfyAudioDSPExciterEnhancer": "Exciter / Enhancer",
    "ComfyAudioDSPPitchShifter": "Pitch Shifter",
    "ComfyAudioDSPTimeStretcher": "Time Stretcher",
    "ComfyAudioDSPResamplerClassic": "Resampler (Classic)",
    "ComfyAudioDSPHarmonizer": "Harmonizer",
    "ComfyAudioDSPPitchCorrection": "Pitch Correction (Auto-Tune style)",
    "ComfyAudioDSPVarispeedPlayer": "Varispeed Player",
    "ComfyAudioDSPPannerBalance": "Panner (Balance)",
    "ComfyAudioDSPStereoWidth": "Stereo Width",
    "ComfyAudioDSPMidSideEncoder": "Mid/Side Encoder",
    "ComfyAudioDSPMidSideDecoder": "Mid/Side Decoder",
    "ComfyAudioDSPMidSideEQ": "Mid/Side EQ",
    "ComfyAudioDSPStereoEnhancerHaas": "Stereo Enhancer / Haas Effect",
    "ComfyAudioDSPSwapChannels": "Swap Channels",
    "ComfyAudioDSPMonoMaker": "Mono Maker",
    "ComfyAudioDSPBinauralPanner": "Binaural Panner (HRTF)",
    "ComfyAudioDSPHRTFConvolution": "HRTF Convolution",
    "ComfyAudioDSPAmbisonicsEncoder": "Ambisonics Encoder (1st order)",
    "ComfyAudioDSPAmbisonicsDecoder": "Ambisonics Decoder (Stereo/Binaural)",
    "ComfyAudioDSPAmbisonicsRotator": "Ambisonics Rotator",
    "ComfyAudioDSPDistanceSimulator": "Distance Simulator",
    "ComfyAudioDSPDopplerEffect": "Doppler Effect",
    "ComfyAudioDSPRMSMeter": "RMS Meter",
    "ComfyAudioDSPPeakMeter": "Peak Meter",
    "ComfyAudioDSPLUFSMeter": "LUFS Meter",
    "ComfyAudioDSPLoudnessGraph": "Loudness Graph",
    "ComfyAudioDSPSpectralAnalyzer": "Spectral Analyzer (FFT)",
    "ComfyAudioDSPSpectrogramVisualizer": "Spectrogram Visualizer",
    "ComfyAudioDSPWaveformVisualizer": "Waveform Visualizer",
    "ComfyAudioDSPPhaseCorrelationMeter": "Phase Correlation Meter",
    "ComfyAudioDSPGoniometerVectorscope": "Goniometer / Vectorscope",
    "ComfyAudioDSPBPMTempoDetector": "BPM / Tempo Detector",
    "ComfyAudioDSPKeyPitchDetector": "Key / Pitch Detector",
    "ComfyAudioDSPTransientOnsetDetector": "Transient / Onset Detector",
    "ComfyAudioDSPSilenceDetector": "Silence Detector",
    "ComfyAudioDSPSineWaveGenerator": "Sine Wave Generator",
    "ComfyAudioDSPNoiseGenerator": "White / Pink / Brown Noise",
    "ComfyAudioDSPSweepChirp": "Sweep / Chirp",
    "ComfyAudioDSPImpulse": "Impulse",
    "ComfyAudioDSPOscillatorMultiWave": "Oscillator (Multi-wave)",
    "ComfyAudioDSPClickTrackMetronome": "Click Track / Metronome",
    "ComfyAudioDSPAudioMixer": "Audio Mixer (N channels)",
    "ComfyAudioDSPAudioRouterSelector": "Audio Router / Selector",
    "ComfyAudioDSPAudioSplitter": "Audio Splitter",
    "ComfyAudioDSPAudioMerger": "Audio Merger",
    "ComfyAudioDSPCrossfader": "Crossfader",
    "ComfyAudioDSPSidechainGateCompressor": "Sidechain Gate / Compressor",
    "ComfyAudioDSPSendReturnLoop": "Send/Return Loop",
    "ComfyAudioDSPParallelProcessingRouter": "Parallel Processing Router",
    "ComfyAudioDSPParallelReturnMixer": "Parallel Return Mixer",
    "ComfyAudioDSPGainTrim": "Gain / Trim",
    "ComfyAudioDSPPhaseInverter": "Phase Inverter",
    "ComfyAudioDSPDCOffsetRemover": "DC Offset Remover",
    "ComfyAudioDSPFadeInOut": "Fade In / Fade Out",
    "ComfyAudioDSPAudioTrimCrop": "Audio Trim / Crop",
    "ComfyAudioDSPSilenceTrimmer": "Silence Trimmer",
    "ComfyAudioDSPNormalize": "Normalize (Peak / RMS / LUFS)",
    "ComfyAudioDSPResampleChangeSampleRate": "Resample / Change Sample Rate",
    "ComfyAudioDSPFormatConverter": "Format Converter (Mono/Stereo)",
    "ComfyAudioDSPAudioInfo": "Audio Info",
    "ComfyAudioDSPDelayCompensation": "Delay Compensation",
    "ComfyAudioDSPLoopDuplicator": "Loop / Duplicator",
    "ComfyAudioDSPReverseAudio": "Reverse Audio",
    "ComfyAudioDSPDither": "Dither",
    "ComfyAudioDSPDynamicEQ": "Dynamic EQ",
    "ComfyAudioDSPVocoder": "Vocoder",
    "ComfyAudioDSPEnvelopeFollowerOutput": "Envelope Follower Output",
    "ComfyAudioDSPMultibandCrossover": "Multiband Crossover",
    "ComfyAudioDSPDeclickDecrackle": "Declick / Decrackle",
    "ComfyAudioDSPSpectralSmoothingContrast": "Spectral Smoothing / Contrast",
    "ComfyAudioDSPHumRemover": "Hum Remover",
    "ComfyAudioDSPPhaseRotatorAllpass": "Phase Rotator / All-Pass Filter",
    "ComfyAudioDSPGranularProcessor": "Granular Processor",
    "ComfyAudioDSPMathSignalMixer": "Math / Signal Mixer",
    "ComfyAudioDSPUpwardCompressor": "Upward Compressor",
    "ComfyAudioDSPParallelCompressionMix": "Parallel Compression Mix",
    "ComfyAudioDSPLinearPhaseEQ": "Linear Phase EQ",
    "ComfyAudioDSPCombFilter": "Comb Filter",
    "ComfyAudioDSPShimmerReverb": "Shimmer Reverb",
    "ComfyAudioDSPReverseDelay": "Reverse Delay",
    "ComfyAudioDSPGranularDelay": "Granular Delay",
    "ComfyAudioDSPSlapEcho": "Slap Echo",
    "ComfyAudioDSPBarberpoleFlanger": "Barberpole Flanger",
    "ComfyAudioDSPAutoFilter": "Auto-Filter",
    "ComfyAudioDSPAutoWah": "Auto-Wah / Envelope Filter",
    "ComfyAudioDSPRhythmicGateStutter": "Rhythmic Gate / Stutter Sequencer",
    "ComfyAudioDSPFoldClip": "Fold & Clip",
    "ComfyAudioDSPAmpSimulator": "Amp Simulator",
    "ComfyAudioDSPCrossoverDistortion": "Crossover Distortion",
    "ComfyAudioDSPFormantShifter": "Formant Shifter",
    "ComfyAudioDSPPSOLAPitchShifter": "PSOLA Pitch Shifter",
    "ComfyAudioDSPPolyphonicPitchCorrection": "Polyphonic Pitch Correction",
    "ComfyAudioDSPTruePeakMeter": "True Peak Meter",
    "ComfyAudioDSPDynamicRangeDRMeter": "Dynamic Range DR Meter",
    "ComfyAudioDSPVBAPPanner": "VBAP Panner",
    "ComfyAudioDSPHigherOrderAmbisonicsEncoder": "Higher-Order Ambisonics Encoder",
    "ComfyAudioDSPHigherOrderAmbisonicsDecoder": "Higher-Order Ambisonics Decoder",
    "ComfyAudioDSPHigherOrderAmbisonicsRotator": "Higher-Order Ambisonics Rotator",
    "ComfyAudioDSPSixDOFRenderer": "6DOF Renderer",
    "ComfyAudioDSPBRIRConvolution": "BRIR Convolution",
    "ComfyAudioDSPFMOperator": "FM Operator",
    "ComfyAudioDSPKarplusStrongString": "Karplus-Strong String",
    "ComfyAudioDSPWavetableOscillator": "Wavetable Oscillator",
    "ComfyAudioDSPSamplePlayer": "Sample Player",
    "ComfyAudioDSPSpectralGate": "Spectral Gate",
    "ComfyAudioDSPSpectralFreeze": "Spectral Freeze",
    "ComfyAudioDSPFrequencyShifter": "Frequency Shifter",
    "ComfyAudioDSPSpectralBlur": "Spectral Blur",
    "ComfyAudioDSPSpectralNoiseReduction": "Spectral Noise Reduction",
    "ComfyAudioDSPLFOSource": "LFO Source",
    "ComfyAudioDSPADSREnvelopeGenerator": "ADSR Envelope Generator",
    "ComfyAudioDSPSampleAndHold": "Sample & Hold",
    "ComfyAudioDSPStepSequencer": "Step Sequencer",
    "ComfyAudioDSPDeClip": "De-Clip",
    "ComfyAudioDSPDeReverb": "De-Reverb",
    "ComfyAudioDSPBandwidthExtension": "Bandwidth Extension",
    "ComfyAudioDSPDenoiser": "Denoiser",
    "ComfyAudioDSPAudioFeatureToText": "Audio Feature to Text",
    "ComfyAudioDSPBeatSlicer": "Beat Slicer",
    "ComfyAudioDSPAudioQualityEstimator": "Audio Quality Estimator",
    "ComfyAudioDSPSongAnalysisToDSPControls": "Song Analysis to DSP Controls",
    "ComfyAudioDSPSongKeyToPitchControls": "Song Key to Pitch Controls",
    "ComfyAudioDSPSongSegmentSelector": "Song Segment Selector",
    "ComfyAudioDSPSongBeatGridSlicer": "Song Beat Grid Slicer",
}

NODE_DISPLAY_NAME_MAPPINGS = {key: loc.display_name(key, value) for key, value in _DISPLAY_FALLBACKS.items()}
