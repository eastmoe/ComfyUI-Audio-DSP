# ComfyUI Audio DSP

ComfyUI custom nodes for practical audio DSP. Nodes use the native ComfyUI
`AUDIO` type:

```python
{"waveform": torch.Tensor[B, C, T], "sample_rate": int}
```

No extra runtime packages are required beyond Python, ComfyUI, and its existing
audio/scientific dependencies (`torch`, `torchaudio`, `numpy`, `scipy`).

## Menu

Right-click menu:

`eastmoe -> Comfy-Audio-DSP -> Dynamics`
`eastmoe -> Comfy-Audio-DSP -> Equalizers & Filters`
`eastmoe -> Comfy-Audio-DSP -> Reverb`
`eastmoe -> Comfy-Audio-DSP -> Delay & Echo`
`eastmoe -> Comfy-Audio-DSP -> Modulation`
`eastmoe -> Comfy-Audio-DSP -> Distortion & Saturation`
`eastmoe -> Comfy-Audio-DSP -> Pitch & Time`
`eastmoe -> Comfy-Audio-DSP -> Stereo Imaging`
`eastmoe -> Comfy-Audio-DSP -> Spatial & 3D`
`eastmoe -> Comfy-Audio-DSP -> Metering & Analysis`
`eastmoe -> Comfy-Audio-DSP -> Signal Generators`
`eastmoe -> Comfy-Audio-DSP -> Routing & Mixing`
`eastmoe -> Comfy-Audio-DSP -> Utilities`

## Dynamics Nodes

| Node | Summary |
| --- | --- |
| Compressor | Threshold, ratio, attack, release, knee, makeup gain, detector, mix. |
| Limiter | Brickwall-style limiter with threshold, release, and lookahead. |
| Noise Gate | Threshold, attack, hold, release, and attenuation range. |
| Expander | Downward expander for reducing low-level signal. |
| Transient Shaper | Independent attack and sustain gain controls. |
| De-Esser | Band-limited compressor for reducing sibilance, typically 4-10 kHz. |
| Multi-band Compressor | 3 or 4 bands with per-band threshold, ratio, and makeup gain. |
| Auto Gain / Leveler | RMS or peak target level matching with smoothed gain. |
| Loudness Normalizer (LUFS) | Integrated, short-term, or momentary LUFS normalization. |
| Upward Compressor | Raises low-level material below a threshold. |
| Parallel Compression Mix | Dry/compressed blend for parallel compression. |

## Equalizers & Filters Nodes

| Node | Summary |
| --- | --- |
| Low Shelf / High Shelf | Shelf EQ with frequency, gain, Q/slope, and mix. |
| Peak / Bell Filter | Parametric peak/bell EQ with frequency, gain, and Q. |
| Low Pass / High Pass Filter | Low-pass or high-pass filter with cutoff, order, and Q. |
| Band Pass / Band Stop Filter | Band-pass or band-stop filter with center frequency and bandwidth. |
| Notch Filter | Very narrow band-stop filter for removing a single frequency. |
| 3-Band EQ | Low shelf, mid peak, and high shelf in one classic EQ. |
| Parametric EQ (N bands) | Up to 8 configurable bands with selectable filter type. |
| Graphic EQ (10/15/31 bands) | Fixed-frequency graphic EQ with 10, 15, or 31 bands. |
| Tilt EQ | Tilts around a pivot frequency, raising highs while lowering lows or the reverse. |
| RIAA EQ | RIAA phono de-emphasis/playback or pre-emphasis. |
| High Pass / Low Pass with Resonance | Resonant synth-style high-pass or low-pass filter with optional drive. |
| Dynamic EQ | Single-band threshold/ratio-controlled EQ for dynamic resonance cuts or boosts. |
| Spectral Smoothing / Contrast | STFT spectral envelope smoothing or contrast enhancement. |
| Hum Remover | Auto or fixed 50/60 Hz hum removal with harmonic notch filters. |
| Linear Phase EQ | Zero-phase FFT EQ for linear-phase-style magnitude shaping. |
| Comb Filter | Feedforward/feedback comb filtering. |

## Reverb Nodes

| Node | Summary |
| --- | --- |
| Convolution Reverb | WAV impulse-response convolution with dry/wet controls. |
| Schroeder Reverb | Classic comb/all-pass algorithm with pre-delay, decay, diffusion, and tone filters. |
| Freeverb / Moorer Reverb | Feedback delay network style reverb with damping, stereo width, and early reflections. |
| Spring Reverb Sim | All-pass and delay-network spring reverb approximation. |
| Plate Reverb Sim | Dense plate-style diffusion and smooth decay. |
| Gated Reverb | Nonlinear gated tail cutoff for 1980s-style ambience. |
| Reverse Reverb | Reverse audio, apply reverb, then reverse back for swelling effects. |
| Shimmer Reverb | Reverb with octave-shifted shimmer in the tail. |

## Delay & Echo Nodes

| Node | Summary |
| --- | --- |
| Simple Delay | Basic delay with time, feedback, and dry/wet mix. |
| Tempo-synced Delay | BPM and note-value delay times, including dotted and triplet values. |
| Ping-Pong Delay | Alternating left/right feedback delay. |
| Multi-tap Delay | Six independent delay taps with time and gain controls. |
| Dub Delay | Tape/BBD-style delay with low-pass tone and wow/flutter. |
| Filtered Delay | Delay with high-pass and low-pass filtering. |
| Stereo Spread Delay | Small left/right delay offsets for stereo width. |
| Reverse Delay | Reverse-style delayed repeats. |
| Granular Delay | Delayed wet path broken into grains. |
| Slap Echo | Slapback echo presets. |

## Modulation Nodes

| Node | Summary |
| --- | --- |
| Chorus | Multi-voice modulated delay with depth, rate, feedback, and mix. |
| Flanger | Short LFO-modulated delay with feedback. |
| Phaser | Cascaded all-pass phaser with selectable stages. |
| Tremolo | Amplitude LFO modulation with sine, triangle, or square waves. |
| Vibrato | Small pitch modulation using fractional delay. |
| Rotary Speaker (Leslie) | Split-band rotary speaker approximation with low/high speeds. |
| Ring Modulator | Carrier oscillator multiplication for metallic sidebands. |
| Auto Panner | LFO-controlled stereo panning. |
| Uni-vibe | Vintage optical phaser/chorus-style modulation. |
| Vocoder | Filter-bank vocoder using a modulator envelope to shape a carrier. |
| Barberpole Flanger | Continuously rising or falling flanger illusion. |
| Auto-Filter | LFO-controlled filter sweep. |

## Distortion & Saturation Nodes

| Node | Summary |
| --- | --- |
| Soft Clipper | tanh or cubic soft clipping. |
| Hard Clipper | Threshold-based hard clipping. |
| Tube Saturation | Asymmetric waveshaping for even harmonics. |
| Tape Saturation | Tape-style drive, compression, tone shaping, and wow. |
| Fuzz | Aggressive square-like clipping. |
| Bit Crusher | Lower bit depth and sample-rate resolution. |
| Overdrive / Distortion | Drive, tone, mode, output gain, and mix. |
| Wavefolder | Wavefolding for complex harmonics. |
| Exciter / Enhancer | Generates high-frequency harmonics for brightness. |
| Fold & Clip | Hybrid wavefolding and clipping distortion. |
| Amp Simulator | Simple amp and cabinet simulator. |

## Pitch & Time Nodes

| Node | Summary |
| --- | --- |
| Pitch Shifter | Shifts pitch by semitones and cents while keeping duration. |
| Time Stretcher | Changes duration from 0.5x to 2x while preserving pitch. |
| Resampler (Classic) | Tape-style resampling where speed, pitch, and duration change together. |
| Harmonizer | Generates up to four fixed-interval harmony voices. |
| Pitch Correction (Auto-Tune style) | Monophonic pitch correction to a selected key and scale. |
| Varispeed Player | Varispeed playback with linked speed, pitch, and duration. |
| Granular Processor | Windowed grains with pitch, jitter, scatter, and reverse probability. |
| Formant Shifter | Spectral-envelope/formant shifting. |
| Polyphonic Pitch Correction | Spectral pitch-class correction for chords and mixes. |

## Stereo Imaging Nodes

| Node | Summary |
| --- | --- |
| Panner (Balance) | Left/right balance with optional equal-power law. |
| Stereo Width | Mid/side stereo width expansion or narrowing. |
| Mid/Side Encoder | Encodes LR stereo to Mid and Side channels. |
| Mid/Side Decoder | Decodes Mid and Side channels back to LR stereo. |
| Mid/Side EQ | Independent EQ/filtering for Mid and Side. |
| Stereo Enhancer / Haas Effect | Short channel delay for Haas stereo widening. |
| Swap Channels | Swaps left and right channels. |
| Mono Maker | Makes low frequencies below a cutoff mono. |

## Spatial & 3D Nodes

| Node | Summary |
| --- | --- |
| Binaural Panner (HRTF) | Built-in ITD/ILD binaural panner for mono sources. |
| HRTF Convolution | SOFA HRTF convolution using the nearest measured direction. |
| Ambisonics Encoder (1st order) | Encodes mono audio to WXYZ first-order Ambisonics. |
| Ambisonics Decoder (Stereo/Binaural) | Decodes first-order Ambisonics to stereo or simple binaural output. |
| Ambisonics Rotator | Rotates WXYZ soundfields by yaw, pitch, and roll. |
| Distance Simulator | Distance gain, air absorption, predelay, and reverb blend. |
| Doppler Effect | Doppler pitch and delay from changing distance and velocity. |
| VBAP Panner | Multi-speaker panning from speaker angles. |
| Higher-Order Ambisonics Encoder | Approximate HOA channel encoder. |

## Metering & Analysis Nodes

| Node | Summary |
| --- | --- |
| RMS Meter | RMS level in dBFS with channel details. |
| Peak Meter | Peak level, overload flag, and channel details. |
| LUFS Meter | Approximate integrated, short-term, momentary LUFS and LRA. |
| Spectral Analyzer (FFT) | FFT spectrum image plus sampled bin data. |
| Spectrogram Visualizer | Spectrogram image for preview nodes. |
| Waveform Visualizer | Waveform image for preview nodes. |
| Phase Correlation Meter | Stereo correlation from -1 to +1. |
| Goniometer / Vectorscope | Lissajous vectorscope image. |
| BPM / Tempo Detector | BPM estimate and beat time list. |
| Key / Pitch Detector | Dominant pitch and nearest pitch class. |
| Transient / Onset Detector | Onset time list. |
| Silence Detector | Silent range list below a threshold. |
| True Peak Meter | Oversampled true-peak dBFS and overload flag. |
| Dynamic Range DR Meter | Crest-factor-style DR estimate. |

## Signal Generators Nodes

| Node | Summary |
| --- | --- |
| Sine Wave Generator | Frequency, amplitude, duration, sample rate, channels. |
| White / Pink / Brown Noise | Test noise with type, amplitude, duration, and seed. |
| Sweep / Chirp | Linear or logarithmic sweep. |
| Impulse | Single-sample impulse or short click. |
| Oscillator (Multi-wave) | Sine, triangle, saw, or square oscillator with duty cycle. |
| Click Track / Metronome | BPM, time signature, and bar count click track. |
| FM Operator | Carrier/modulator FM tone generator. |
| Karplus-Strong String | Plucked-string physical modeling. |
| Wavetable Oscillator | Built-in wavetable oscillator. |
| Sample Player | Loads, trims, loops, gains, and resamples WAV samples. |

## Routing & Mixing Nodes

| Node | Summary |
| --- | --- |
| Audio Mixer (N channels) | Up to 8 inputs with gain, pan, mute, solo, and master gain. |
| Audio Router / Selector | Selects one input by index. |
| Audio Splitter | Splits up to four mono channel outputs. |
| Audio Merger | Merges mono inputs to stereo or multichannel audio. |
| Crossfader | Equal-power or linear crossfade between two inputs. |
| Sidechain Gate / Compressor | Uses external sidechain key input. |
| Send/Return Loop | Dry/return blend for effect-send workflows. |
| Multiband Crossover | Splits audio into 3 or 4 frequency bands for parallel processing. |

## Utilities Nodes

| Node | Summary |
| --- | --- |
| Gain / Trim | Simple dB gain. |
| Phase Inverter | Polarity inversion for mono, left, and/or right. |
| DC Offset Remover | Mean removal or high-pass DC filtering. |
| Fade In / Fade Out | Linear, exponential, or S-curve fades. |
| Audio Trim / Crop | Time-based crop. |
| Silence Trimmer | Automatic leading/trailing silence trim. |
| Normalize (Peak / RMS / LUFS) | Peak, RMS, or approximate LUFS normalization. |
| Resample / Change Sample Rate | High-quality polyphase resampling. |
| Format Converter (Mono/Stereo) | Mono mix, channel select, or stereo duplicate. |
| Audio Info | Sample rate, duration, channels, samples, and metadata text. |
| Delay Compensation | Adds manual sample/ms delay. |
| Loop / Duplicator | Repeats by loop count or target duration. |
| Reverse Audio | Reverses playback direction. |
| Envelope Follower Output | Attack/release envelope as scalar, text points, and audio-rate control signal. |
| Declick / Decrackle | Median-based repair for short impulse clicks and crackle. |
| Phase Rotator / All-Pass Filter | Adjustable all-pass phase rotation without intentional magnitude change. |
| Math / Signal Mixer | Safe sample-wise formula over up to four audio inputs. |

## Repository Layout

```text
src/comfy_audio_dsp/  Node registration, shared helpers, and categorized DSP modules
doc/                  Documentation
local/zh-cn/          Chinese node localization loaded at runtime
temp/                 Runtime scratch directory
static/               Optional frontend assets
```

## Notes

The LUFS node implements an internal BS.1770/EBU R128-style approximation using
K-weighting, absolute/relative gating for integrated loudness, and true-peak
ceiling protection by batch peak limiting.

The DSP implementation is split by category:

- `common.py`: shared audio tensor, filtering, metering, and mix helpers.
- `analysis.py`: metering values, detection helpers, and generated preview images.
- `dynamics.py`: compressor, limiter, gate, expander, de-esser, multiband, gain, and loudness processors.
- `delay.py`: delay and echo processors.
- `equalizers.py`: equalizer and filter processors.
- `generators.py`: test tones, noise, sweeps, impulses, oscillators, and click tracks.
- `modulation.py`: chorus, flanger, phaser, tremolo, vibrato, rotary, ring modulation, panning, and Uni-vibe processors.
- `pitch_time.py`: pitch shifting, time stretching, harmonizing, correction, classic resampling, and varispeed processors.
- `reverb.py`: convolution, algorithmic, gated, and reverse reverbs.
- `routing.py`: mixing, selection, splitting, merging, crossfading, sidechain, and send/return processors.
- `saturation.py`: clipping, saturation, fuzz, bit crushing, overdrive, wavefolding, and exciter processors.
- `stereo.py`: panning, stereo width, Mid/Side tools, Haas widening, channel swap, and mono-maker processors.
- `spatial.py`: binaural panning, optional SOFA HRTF convolution, first-order Ambisonics, distance, and Doppler processors.
- `utilities.py`: gain, polarity, DC removal, fades, trim, normalize, resample, format, info, delay, loop, and reverse tools.
