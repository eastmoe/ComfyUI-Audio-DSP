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
`eastmoe -> Comfy-Audio-DSP -> 音调与时间变换`
`eastmoe -> Comfy-Audio-DSP -> 立体声场控制`
`eastmoe -> Comfy-Audio-DSP -> 空间音频`
`eastmoe -> Comfy-Audio-DSP -> 测量与分析`
`eastmoe -> Comfy-Audio-DSP -> 信号发生器`
`eastmoe -> Comfy-Audio-DSP -> 路由与混音`
`eastmoe -> Comfy-Audio-DSP -> 实用工具`
`eastmoe -> Comfy-Audio-DSP -> 频谱域处理`
`eastmoe -> Comfy-Audio-DSP -> 模块化调制源`
`eastmoe -> Comfy-Audio-DSP -> 音频修复`
`eastmoe -> Comfy-Audio-DSP -> 工作流集成`

## 动态处理节点

| 节点 | 功能 |
| --- | --- |
| Compressor | 压缩器：阈值、比率、启动、释放、膝宽、增益补偿、检测方式、干湿混合。 |
| Limiter | 砖墙限制器：阈值、释放时间、前瞻，防止削波。 |
| M/S Compressor | M/S 压缩器：分别压缩 Mid 与 Side。 |
| Noise Gate | 噪声门：阈值、启动、保持、释放、衰减范围。 |
| Expander | 向下扩展器：降低低电平信号。 |
| Transient Shaper | 瞬态塑形器：独立控制 Attack 和 Sustain 的增益。 |
| De-Esser | 嘶声消除器：指定频段压缩，常用于 4-10 kHz。 |
| Multi-band Compressor | 多段压缩：支持 3 或 4 段，各段有独立阈值、比率和补偿增益。 |
| Multi-band Limiter | 多段限制器：分频后逐段限制并安全求和。 |
| Auto Gain / Leveler | 自动增益：按 RMS 或峰值匹配目标电平。 |
| Loudness Normalizer (LUFS) | 响度标准化：支持整体、短时、瞬时 LUFS 归一化。 |
| Upward Compressor | 向上压缩器：提升低于阈值的低电平内容。 |
| Parallel Compression Mix | 并行压缩混合：干声与压缩通路混合。 |

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
| Dynamic EQ | 动态 EQ：单频段阈值/比率控制，用于动态削共振或增强基频。 |
| Spectral Smoothing / Contrast | 频谱抹平/对比增强：平滑或增强频谱包络。 |
| Hum Remover | 去嗡嗡声：自动或固定 50/60 Hz 及谐波陷波。 |
| Match EQ | 频谱匹配均衡：将当前素材频谱包络匹配到参考音频。 |
| Linear Phase EQ | 线性相位 EQ：零相位 FFT 幅度塑形。 |
| Comb Filter | 梳状滤波器：前馈/反馈梳状滤波。 |

## 频谱域处理节点

| 节点 | 功能 |
| --- | --- |
| Spectral Gate | 频谱门：基于 FFT bin 的噪声门。 |
| Spectral Freeze | 频谱冻结：冻结指定 FFT 帧生成持续音色。 |
| Frequency Shifter | 线性频移：不同于变调，整体平移频率。 |
| Spectral Blur | 频谱模糊：在频率/时间轴上模糊音色。 |
| Spectral Noise Reduction | 频谱降噪：根据开头噪声轮廓做频谱减法。 |

## 混响节点

| 节点 | 功能 |
| --- | --- |
| Convolution Reverb | 卷积混响：加载 WAV 脉冲响应，调节干声/湿声。 |
| IR Manager | IR 管理器：加载、裁剪、反转、标准化 IR WAV。 |
| Schroeder Reverb | Schroeder 算法混响：预延迟、衰减时间、扩散度、高低切。 |
| Freeverb / Moorer Reverb | 反馈延迟网络风格混响：房间尺寸、阻尼、立体声宽度、早期反射。 |
| Spring Reverb Sim | 弹簧混响模拟：串联全通滤波器和延迟网络。 |
| Plate Reverb Sim | 板式混响模拟：密集扩散和平滑衰减。 |
| Gated Reverb | 门控混响：非线性尾音截断，80 年代风格。 |
| Reverse Reverb | 反向混响：反转音频、施加混响、再反转，制造膨胀效果。 |
| Shimmer Reverb | Shimmer 混响：混响尾音中加入八度移调成分。 |
| FDN Reverb | FDN 反馈延迟网络混响：扩散、阻尼和调制。 |

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
| Reverse Delay | 反向延迟：倒放式延迟重复。 |
| Granular Delay | 粒子延迟：将延迟湿声拆成颗粒。 |
| Slap Echo | Slap Echo：常用 slapback 回声预设。 |
| Echoplex Tape Echo | Echoplex 磁带回声：饱和、老化、哇音和反馈。 |

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
| Vocoder | 声码器：用调制信号包络塑形载波滤波器组。 |
| Barberpole Flanger | Barberpole 镶边：连续上升或下降的无限镶边错觉。 |
| Auto-Filter | 自动滤波器：LFO 控制滤波器扫动。 |
| Auto-Wah / Envelope Filter | Auto-Wah / 包络滤波器：输入包络控制带通中心频率。 |
| Rhythmic Gate / Stutter Sequencer | 节奏门控 / Stutter 序列器：按节拍图案门控或重复。 |

## 模块化调制源节点

| 节点 | 功能 |
| --- | --- |
| LFO Source | 独立 LFO：输出可共享的音频速率控制信号。 |
| ADSR Envelope Generator | ADSR 包络发生器：用于参数自动化。 |
| Sample & Hold | 采样保持：随机步进调制源。 |
| Step Sequencer | 步进序列器：按 BPM 输出参数序列。 |

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
| Fold & Clip | 折叠/削波混合：波折叠与削波混合失真。 |
| Amp Simulator | 音箱模拟器：简单放大器和箱体滤波模拟。 |
| Crossover Distortion | 交越失真：模拟功放死区非线性。 |

## 音调与时间变换节点

| 节点 | 功能 |
| --- | --- |
| Pitch Shifter | 变调器：按半音和音分移动音高，并尽量保持时长。 |
| Time Stretcher | 时间伸缩：在 0.5x-2x 范围改变时长，并尽量保持音高。 |
| Resampler (Classic) | 经典重采样：磁带式变速，速度、音高和时长同时变化。 |
| Harmonizer | 和声器：生成最多 4 个固定音程声部。 |
| Pitch Correction (Auto-Tune style) | 音高修正：将单声部音高拉向指定调性和音阶。 |
| Varispeed Player | 变速播放：模拟设备变速，音高和时长一起改变。 |
| Granular Processor | 颗粒处理器：窗口颗粒重放，支持音高、抖动、散布和反转概率。 |
| Formant Shifter | 共振峰移位器：移动频谱包络。 |
| Polyphonic Pitch Correction | 多声部音高校正：频谱音级约束。 |
| PSOLA Pitch Shifter | PSOLA 变调器：时间域单声部变调选项。 |

## 立体声场控制节点

| 节点 | 功能 |
| --- | --- |
| Panner (Balance) | 平衡控制：左右声道增益调节，可选等功率法则。 |
| Stereo Width | 立体声宽度：通过 Mid/Side 处理扩大或缩小声像。 |
| Mid/Side Encoder | 将 LR 信号编码为 Mid 和 Side。 |
| Mid/Side Decoder | 将 Mid/Side 信号解码回 LR。 |
| Mid/Side EQ | 分别对 Mid 与 Side 应用均衡/滤波。 |
| Stereo Enhancer / Haas Effect | 使用短通道延迟制造 Haas 立体声增强。 |
| Swap Channels | 交换左右声道。 |
| Mono Maker | 将指定截止频率以下的低频强制为单声道。 |

## 空间音频节点

| 节点 | 功能 |
| --- | --- |
| Binaural Panner (HRTF) | 双耳声像器：使用内置 ITD/ILD 近似线索摆放单声源。 |
| HRTF Convolution | 加载 SOFA HRTF 并按最近方向进行双耳卷积。 |
| Ambisonics Encoder (1st order) | 一阶 Ambisonics 编码：单声道输入输出 WXYZ。 |
| Ambisonics Decoder (Stereo/Binaural) | 将一阶 Ambisonics 解码为立体声或简单双耳输出。 |
| Ambisonics Rotator | 旋转 WXYZ Ambisonics 声场。 |
| Distance Simulator | 距离衰减、空气吸收、预延迟和混响比例联动。 |
| Doppler Effect | 根据距离变化和速度模拟多普勒音高/延迟变化。 |
| VBAP Panner | VBAP 摆位：按扬声器角度输出多声道摆位。 |
| Higher-Order Ambisonics Encoder | 高阶 Ambisonics：近似 HOA 多声道编码。 |
| Higher-Order Ambisonics Decoder | 高阶 Ambisonics 解码：输出立体声或自定义扬声器布局。 |
| Higher-Order Ambisonics Rotator | 高阶 Ambisonics 旋转：水平旋转 HOA 声场。 |
| 6DOF Renderer | 6DOF 渲染器：声源/听者位置和朝向双耳渲染。 |
| BRIR Convolution | BRIR 卷积：加载立体声 BRIR WAV 做双耳房间渲染。 |

## 测量与分析节点

| 节点 | 功能 |
| --- | --- |
| RMS Meter | 显示 RMS 电平 dBFS 和通道明细。 |
| Peak Meter | 显示峰值电平、过载状态和通道明细。 |
| LUFS Meter | 近似整体、短时、瞬时 LUFS 和响度范围 LRA。 |
| Spectral Analyzer (FFT) | 输出 FFT 频谱图片和采样频点数据。 |
| Spectrogram Visualizer | 生成可连接预览节点的频谱图图片。 |
| Waveform Visualizer | 生成可连接预览节点的波形图图片。 |
| Phase Correlation Meter | 输出 -1 到 +1 的立体声相位相关性。 |
| Goniometer / Vectorscope | 生成李萨如/矢量示波器图片。 |
| BPM / Tempo Detector | 估算 BPM 并输出节拍时间列表。 |
| Key / Pitch Detector | 估算主频率和最近音名。 |
| Transient / Onset Detector | 检测起始点时间列表。 |
| Silence Detector | 检测低于阈值的静音段落。 |
| True Peak Meter | 真峰值表：过采样峰值 dBFS 和过载状态。 |
| Dynamic Range DR Meter | 动态范围 DR 表：基于峰值/RMS 的 DR 估算。 |

## 信号发生器节点

| 节点 | 功能 |
| --- | --- |
| Sine Wave Generator | 正弦波：频率、振幅、时长、采样率、声道数。 |
| White / Pink / Brown Noise | 白噪声、粉噪声或棕噪声测试信号。 |
| Sweep / Chirp | 线性或对数扫频信号。 |
| Impulse | 单采样脉冲或短点击音。 |
| Oscillator (Multi-wave) | 正弦、三角、锯齿、方波，可调占空比。 |
| Click Track / Metronome | 按 BPM、拍号和小节数生成节拍器。 |
| FM Operator | FM 算子：载波/调制器 FM 音源。 |
| Karplus-Strong String | Karplus-Strong 弦模拟：拨弦物理建模。 |
| Wavetable Oscillator | 波表振荡器：内置波表音源。 |
| Sample Player | 采样播放器：加载、裁剪、循环、增益和重采样 WAV。 |

## 路由与混音节点

| 节点 | 功能 |
| --- | --- |
| Audio Mixer (N channels) | 最多 8 路输入，每路有增益、声像、静音、独奏和主增益。 |
| Audio Router / Selector | 按索引选择一路音频输出。 |
| Audio Splitter | 拆分最多 4 路单声道输出。 |
| Audio Merger | 将多个单声道合并为立体声或多通道。 |
| Crossfader | 在两路输入间线性或等功率交叉淡化。 |
| Sidechain Gate / Compressor | 使用外部侧链输入控制门限或压缩。 |
| Send/Return Loop | 用于效果发送/返回流程的干声与返回信号混合。 |
| Parallel Processing Router | 并联处理路由器：拆分干声与发送路径。 |
| Parallel Return Mixer | 并联返回混合器：回收并联处理返回。 |
| Multiband Crossover | 多频段分频器：拆分为 3 或 4 个频段用于并行处理。 |

## 实用工具节点

| 节点 | 功能 |
| --- | --- |
| Gain / Trim | 简单 dB 增益。 |
| Phase Inverter | 反转单声道、左声道和/或右声道极性。 |
| DC Offset Remover | 均值移除或高通去除直流偏移。 |
| Fade In / Fade Out | 线性、指数或 S 曲线淡入淡出。 |
| Audio Trim / Crop | 按时间裁剪音频片段。 |
| Silence Trimmer | 自动切除首尾静音。 |
| Normalize (Peak / RMS / LUFS) | 峰值、RMS 或近似 LUFS 归一化。 |
| Resample / Change Sample Rate | 高质量多相重采样。 |
| Format Converter (Mono/Stereo) | 单声道混合、声道选择或立体声复制。 |
| Audio Info | 采样率、时长、声道数、采样数和元数据文本。 |
| Delay Compensation | 手动添加采样点或毫秒延迟。 |
| Loop / Duplicator | 按次数循环或重复到目标时长。 |
| Reverse Audio | 反转音频播放方向。 |
| Dither | 抖动：RPDF、TPDF 或简单噪声整形固定点量化。 |
| Envelope Follower Output | 包络跟随器输出：标量、文本点列表和音频速率控制信号。 |
| Declick / Decrackle | 去咔嗒/去噼啪：基于中值的短脉冲修复。 |
| Phase Rotator / All-Pass Filter | 相位旋转/全通滤波：改变相位而尽量不改变幅度。 |
| Math / Signal Mixer | 数学/信号混算：对最多 4 路音频执行安全逐样本公式。 |

## 音频修复节点

| 节点 | 功能 |
| --- | --- |
| De-Clip | 削波修复：插值修复短时过载区域。 |
| De-Reverb | 去混响：用 STFT 尾音抑制减少房间混响。 |
| Bandwidth Extension | 带宽扩展：合成高频泛音补全带宽。 |
| Denoiser | 宽带降噪：频谱减法或维纳式连续底噪抑制。 |

## 工作流集成节点

| 节点 | 功能 |
| --- | --- |
| Audio Feature to Text | 音频特征转文本：输出 BPM、调性、响度和音色描述。 |
| Beat Slicer | 节拍切片器：按瞬态生成切片时间段。 |
| Audio Quality Estimator | 音频质量估算器：输出质量分数和明细。 |

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
- `analysis.py`：电平测量、检测器和预览图片生成。
- `dynamics.py`：压缩、限制、噪声门、扩展、齿音消除、多段压缩、自动增益和响度处理。
- `delay.py`：延迟与回声处理。
- `equalizers.py`：均衡与滤波处理。
- `generators.py`：测试音、噪声、扫频、脉冲、振荡器和节拍器。
- `modulation.py`：合唱、镶边、移相、震音、颤音、旋转扬声器、环形调制、声像和 Uni-vibe。
- `pitch_time.py`：变调、时间伸缩、和声、音高修正、经典重采样和变速播放。
- `reverb.py`：卷积混响、算法混响、门控混响和反向混响。
- `restoration.py`：削波修复、去混响和带宽扩展。
- `routing.py`：混音、选择、拆分、合并、交叉淡化、侧链和发送/返回。
- `saturation.py`：削波、饱和、法兹、比特粉碎、过载、波折和激励器。
- `stereo.py`：声像、立体声宽度、Mid/Side、Haas 增强、声道交换和低频单声道化。
- `spatial.py`：双耳声像、可选 SOFA HRTF、一阶 Ambisonics、距离和多普勒处理。
- `spectral.py`：频谱门、冻结、模糊、频移和频谱降噪。
- `utilities.py`：增益、极性、直流偏移、淡化、裁剪、归一化、重采样、格式、信息、延迟、循环和反转工具。
- `workflow.py`：音频特征文本、节拍切片和质量估算。
