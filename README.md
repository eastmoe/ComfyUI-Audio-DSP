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
- `dynamics.py`: compressor, limiter, gate, expander, de-esser, multiband, gain, and loudness processors.
- `delay.py`: delay and echo processors.
- `equalizers.py`: equalizer and filter processors.
- `modulation.py`: chorus, flanger, phaser, tremolo, vibrato, rotary, ring modulation, panning, and Uni-vibe processors.
- `reverb.py`: convolution, algorithmic, gated, and reverse reverbs.
- `saturation.py`: clipping, saturation, fuzz, bit crushing, overdrive, wavefolding, and exciter processors.
