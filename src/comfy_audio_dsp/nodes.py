from __future__ import annotations

from . import dsp
from . import localization as loc
from .delay import NOTE_VALUES
from .equalizers import FILTER_TYPES, GRAPHIC_EQ_BANDS
from .modulation import WAVEFORMS
from .pitch_time import PITCH_KEYS, PITCH_SCALES
from .spatial import SPATIAL_DECODER_MODES
from .stereo import MID_SIDE_EQ_FILTER_TYPES

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


def _ui(section: str, name: str, fallback: str | None = None) -> dict:
    return loc.ui(section, name, fallback)


def _audio_input(section: str = "common", name: str = "audio") -> tuple:
    return ("AUDIO", _ui(section, name, "audio"))


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


class _AudioDSPNode:
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"

    @classmethod
    def _finish(cls, inputs: dict, section: str) -> dict:
        return {"required": {"audio": _audio_input(section), **inputs}}


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
        }, section)

    def process(self, audio, bpm, note_value, feedback, mix):
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
        }, section)

    def process(self, audio, key, scale, correction_speed, mix):
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
    "ComfyAudioDSPConvolutionReverb": ComfyAudioDSPConvolutionReverb,
    "ComfyAudioDSPSchroederReverb": ComfyAudioDSPSchroederReverb,
    "ComfyAudioDSPFreeverbMoorerReverb": ComfyAudioDSPFreeverbMoorerReverb,
    "ComfyAudioDSPSpringReverb": ComfyAudioDSPSpringReverb,
    "ComfyAudioDSPPlateReverb": ComfyAudioDSPPlateReverb,
    "ComfyAudioDSPGatedReverb": ComfyAudioDSPGatedReverb,
    "ComfyAudioDSPReverseReverb": ComfyAudioDSPReverseReverb,
    "ComfyAudioDSPSimpleDelay": ComfyAudioDSPSimpleDelay,
    "ComfyAudioDSPTempoSyncedDelay": ComfyAudioDSPTempoSyncedDelay,
    "ComfyAudioDSPPingPongDelay": ComfyAudioDSPPingPongDelay,
    "ComfyAudioDSPMultiTapDelay": ComfyAudioDSPMultiTapDelay,
    "ComfyAudioDSPDubDelay": ComfyAudioDSPDubDelay,
    "ComfyAudioDSPFilteredDelay": ComfyAudioDSPFilteredDelay,
    "ComfyAudioDSPStereoSpreadDelay": ComfyAudioDSPStereoSpreadDelay,
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
}

_DISPLAY_FALLBACKS = {
    "ComfyAudioDSPCompressor": "Compressor",
    "ComfyAudioDSPLimiter": "Limiter",
    "ComfyAudioDSPNoiseGate": "Noise Gate",
    "ComfyAudioDSPExpander": "Expander",
    "ComfyAudioDSPTransientShaper": "Transient Shaper",
    "ComfyAudioDSPDeEsser": "De-Esser",
    "ComfyAudioDSPMultiBandCompressor": "Multi-band Compressor",
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
    "ComfyAudioDSPConvolutionReverb": "Convolution Reverb",
    "ComfyAudioDSPSchroederReverb": "Schroeder Reverb",
    "ComfyAudioDSPFreeverbMoorerReverb": "Freeverb / Moorer Reverb",
    "ComfyAudioDSPSpringReverb": "Spring Reverb Sim",
    "ComfyAudioDSPPlateReverb": "Plate Reverb Sim",
    "ComfyAudioDSPGatedReverb": "Gated Reverb",
    "ComfyAudioDSPReverseReverb": "Reverse Reverb",
    "ComfyAudioDSPSimpleDelay": "Simple Delay",
    "ComfyAudioDSPTempoSyncedDelay": "Tempo-synced Delay",
    "ComfyAudioDSPPingPongDelay": "Ping-Pong Delay",
    "ComfyAudioDSPMultiTapDelay": "Multi-tap Delay",
    "ComfyAudioDSPDubDelay": "Dub Delay",
    "ComfyAudioDSPFilteredDelay": "Filtered Delay",
    "ComfyAudioDSPStereoSpreadDelay": "Stereo Spread Delay",
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
}

NODE_DISPLAY_NAME_MAPPINGS = {key: loc.display_name(key, value) for key, value in _DISPLAY_FALLBACKS.items()}
