from __future__ import annotations

from . import dsp
from . import localization as loc
from .equalizers import FILTER_TYPES, GRAPHIC_EQ_BANDS

ROOT_CATEGORY = loc.category("root", "eastmoe/Comfy-Audio-DSP")
CATEGORY_DYNAMICS = f"{ROOT_CATEGORY}/{loc.category('dynamics', 'Dynamics')}"
CATEGORY_EQ = f"{ROOT_CATEGORY}/{loc.category('equalizers_filters', 'Equalizers & Filters')}"
CATEGORY_REVERB = f"{ROOT_CATEGORY}/{loc.category('reverb', 'Reverb')}"


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
}

NODE_DISPLAY_NAME_MAPPINGS = {key: loc.display_name(key, value) for key, value in _DISPLAY_FALLBACKS.items()}
