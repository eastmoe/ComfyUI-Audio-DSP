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

## Repository Layout

```text
src/comfy_audio_dsp/  Node registration and DSP implementations
doc/                  Documentation
local/                Localization files
temp/                 Runtime scratch directory
static/               Optional frontend assets
```

## Notes

The LUFS node implements an internal BS.1770/EBU R128-style approximation using
K-weighting, absolute/relative gating for integrated loudness, and true-peak
ceiling protection by batch peak limiting.
