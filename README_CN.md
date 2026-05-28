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

`eastmoe -> Comfy-Audio-DSP -> 动态处理`
`eastmoe -> Comfy-Audio-DSP -> 均衡与滤波`
`eastmoe -> Comfy-Audio-DSP -> 混响`
`eastmoe -> Comfy-Audio-DSP -> 延迟与回声`
`eastmoe -> Comfy-Audio-DSP -> 调制效果`
`eastmoe -> Comfy-Audio-DSP -> 失真与染色`

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

## 均衡与滤波节点

| 节点 | 功能 |
| --- | --- |
| Low Shelf / High Shelf | 搁架式均衡：频率、增益、Q 值/斜率、干湿混合。 |
| Peak / Bell Filter | 峰值/钟形均衡：中心频率、增益、Q 值。 |
| Low Pass / High Pass Filter | 低通/高通滤波：截止频率、阶数、Q 值。 |
| Band Pass / Band Stop Filter | 带通/带阻滤波：中心频率、带宽、阶数。 |
| Notch Filter | 陷波器：极窄带阻，用于去除单一频率。 |
| 3-Band EQ | 三段均衡：低频搁架、中频峰值、高频搁架。 |
| Parametric EQ (N bands) | 参数均衡：最多 8 段，每段类型可选。 |
| Graphic EQ (10/15/31 bands) | 图形均衡：10、15 或 31 段固定频率推子。 |
| Tilt EQ | 倾斜均衡：以枢轴频率为中心，一端提升另一端降低。 |
| RIAA EQ | RIAA 唱机均衡：播放去加重或刻片预加重。 |
| High Pass / Low Pass with Resonance | 带谐振高通/低通：模拟合成器滤波特性，可加驱动。 |

## 混响节点

| 节点 | 功能 |
| --- | --- |
| Convolution Reverb | 卷积混响：加载 WAV 脉冲响应，调节干声/湿声。 |
| Schroeder Reverb | Schroeder 算法混响：预延迟、衰减时间、扩散度、高低切。 |
| Freeverb / Moorer Reverb | 反馈延迟网络风格混响：房间尺寸、阻尼、立体声宽度、早期反射。 |
| Spring Reverb Sim | 弹簧混响模拟：串联全通滤波器和延迟网络。 |
| Plate Reverb Sim | 板式混响模拟：密集扩散和平滑衰减。 |
| Gated Reverb | 门控混响：非线性尾音截断，80 年代风格。 |
| Reverse Reverb | 反向混响：反转音频、施加混响、再反转，制造膨胀效果。 |

## 延迟与回声节点

| 节点 | 功能 |
| --- | --- |
| Simple Delay | 基础延迟：延迟时间、反馈量、干湿混合。 |
| Tempo-synced Delay | 节拍同步延迟：输入 BPM，选择 1/4、1/8、附点、三连音等时值。 |
| Ping-Pong Delay | 乒乓延迟：左右声道交替反馈。 |
| Multi-tap Delay | 多拍点延迟：6 个独立拍点，每个可设置时间和增益。 |
| Dub Delay | 磁带/BBD 风格 Dub 延迟：低通染色、反馈和哇音颤动。 |
| Filtered Delay | 带滤波延迟：延迟信号带高通/低通滤波。 |
| Stereo Spread Delay | 扩散延迟：左右声道微小时差制造立体声宽度。 |

## 调制效果节点

| 节点 | 功能 |
| --- | --- |
| Chorus | 合唱：多声音轻微延迟调制、深度、速率、反馈。 |
| Flanger | 镶边：短延迟、反馈、LFO 调制延迟时间。 |
| Phaser | 移相器：全通滤波器级联 + LFO，级数可调。 |
| Tremolo | 震音：振幅 LFO 调制，可选正弦、三角、方波。 |
| Vibrato | 颤音：小范围音高/延迟调制。 |
| Rotary Speaker (Leslie) | 旋转扬声器：低音转鼓和高音号角独立速度调制。 |
| Ring Modulator | 环形调制：信号与载波相乘，产生金属质感。 |
| Auto Panner | 自动声像：LFO 调制左右声像位置。 |
| Uni-vibe | 复古光耦合移相/合唱风格效果。 |

## 失真与染色节点

| 节点 | 功能 |
| --- | --- |
| Soft Clipper | 软削波：使用 tanh 或三次曲线模拟温和过载。 |
| Hard Clipper | 硬削波：按阈值直接截幅。 |
| Tube Saturation | 电子管饱和：非对称波形塑形、偶次谐波。 |
| Tape Saturation | 磁带饱和：压缩、频响变化和轻微哇音。 |
| Fuzz | 法兹失真：强烈、近似方形削波。 |
| Bit Crusher | 比特粉碎：降低量化位数和采样保持分辨率。 |
| Overdrive / Distortion | 过载/失真：增益、音色控制、混合。 |
| Wavefolder | 波折器：折叠波形产生复杂谐波。 |
| Exciter / Enhancer | 激励器：生成高频谐波增加亮度。 |

## 目录结构

```text
src/comfy_audio_dsp/  节点注册、公共工具与按分类拆分的 DSP 模块
doc/                  文档
local/zh-cn/          运行时加载的中文节点本地化文件
temp/                 运行时临时目录
static/               可选前端资源
```

## 说明

LUFS 节点内置近似 BS.1770/EBU R128 的实现：包含 K-weighting、整体响度的
绝对/相对门限，以及按批次峰值限制的 ceiling 保护。

DSP 实现已按分类拆分：

- `common.py`：音频张量、滤波、包络、电平和干湿混合等公共工具。
- `dynamics.py`：压缩、限制、噪声门、扩展、齿音消除、多段压缩、自动增益和响度处理。
- `delay.py`：延迟与回声处理。
- `equalizers.py`：均衡与滤波处理。
- `modulation.py`：合唱、镶边、移相、震音、颤音、旋转扬声器、环形调制、声像和 Uni-vibe。
- `reverb.py`：卷积混响、算法混响、门控混响和反向混响。
- `saturation.py`：削波、饱和、法兹、比特粉碎、过载、波折和激励器。
