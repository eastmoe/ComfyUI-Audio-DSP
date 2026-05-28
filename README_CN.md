# ComfyUI Audio DSP

面向 ComfyUI 的实用音频 DSP 自定义节点。节点使用 ComfyUI 原生 `AUDIO`
类型：

```python
{"waveform": torch.Tensor[B, C, T], "sample_rate": int}
```

运行时不额外引入新依赖，复用 Python 标准库、ComfyUI 以及已有音频/科学计算依赖
（`torch`、`torchaudio`、`numpy`、`scipy`）。

## 菜单位置

右键菜单：

`eastmoe -> Comfy-Audio-DSP -> Dynamics`

## 动态处理节点

| 节点 | 功能 |
| --- | --- |
| Compressor | 压缩器：阈值、比率、启动、释放、膝宽、增益补偿、检测方式、干湿混合。 |
| Limiter | 砖墙限制器：阈值、释放时间、前瞻，防止削波。 |
| Noise Gate | 噪声门：阈值、启动、保持、释放、衰减范围。 |
| Expander | 向下扩展器：降低低电平信号。 |
| Transient Shaper | 瞬态塑形器：独立控制 Attack 和 Sustain 的增益。 |
| De-Esser | 嘶声消除器：指定频段压缩，常用于 4-10 kHz。 |
| Multi-band Compressor | 多段压缩：支持 3 或 4 段，各段有独立阈值、比率和补偿增益。 |
| Auto Gain / Leveler | 自动增益：按 RMS 或峰值匹配目标电平。 |
| Loudness Normalizer (LUFS) | 响度标准化：支持整体、短时、瞬时 LUFS 归一化。 |

## 目录结构

```text
src/comfy_audio_dsp/  节点注册与 DSP 实现
doc/                  文档
local/                本地化文件
temp/                 运行时临时目录
static/               可选前端资源
```

## 说明

LUFS 节点内置近似 BS.1770/EBU R128 的实现：包含 K-weighting、整体响度的
绝对/相对门限，以及按批次峰值限制的 ceiling 保护。
