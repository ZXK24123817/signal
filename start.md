# 语音信号分析系统

基于 Streamlit 的语音信号分析 Web 应用，集成说话人声纹识别、TTS 防伪检测、中文语音识别 (ASR) 和时频域特征提取。

## 目录结构

```
signal/
├── app.py                  # Streamlit 主界面
├── signal_system.py        # 核心算法模块
├── requirements.txt        # Python 依赖
├── start.md                # 本文件
├── 项目报告.md              # 课程项目报告
│
├── models/                 # 预训练模型（需自行下载或系统自动拉取）
│   ├── speaker/ecapa/      # ECAPA-TDNN 声纹模型（~85MB）
│   └── asr/damo/.../       # Paraformer ASR 模型（~848MB）
│
├── register_id/            # 说话人注册音频目录
│   ├── 张三/
│   │   ├── sample_001.wav
│   │   ├── sample_002.wav
│   │   └── sample_003.wav
│   └── 李四/
│       ├── audio_01.wav
│       └── audio_02.wav
│
├── input/                  # 待批量分析的音频（可选）
├── data/
│   └── speaker_db.json     # 声纹库（注册后自动生成）
└── test_audio/             # 测试音频（可选）
```

## 环境配置

### 1. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：

| 包 | 用途 |
|---|---|
| `streamlit` | Web 界面框架 |
| `torch` | PyTorch 深度学习框架 |
| `speechbrain` | ECAPA-TDNN 声纹识别模型加载 |
| `funasr` | Paraformer 中文语音识别 |
| `librosa` | 音频处理与时频特征提取 |
| `soundfile` | 音频文件读写 |
| `numpy` | 数值计算 |
| `scipy` | 科学计算（信号处理） |
| `matplotlib` | 图表绘制（波形图、频谱图） |

### 3. 模型下载

**ECAPA-TDNN（声纹识别）** — 首次运行自动从 HuggingFace 下载：
- 来源：`speechbrain/spkrec-ecapa-voxceleb`
- 本地路径：`models/speaker/ecapa/`
- 大小：约 85MB

**Paraformer（中文 ASR）** — 首次运行自动从 ModelScope 下载：
- 来源：`damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- 本地路径：`models/asr/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/`
- 大小：约 848MB

> 如果已经手动下载了模型文件，系统会优先使用本地版本，不会重复下载。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

## 使用指南

### 标签页一：音频输入与分析

**输入方式**（三选一）：
- **上传文件**：选择 WAV / MP3 / FLAC / M4A / OGG 音频文件
- **现场录音**：通过浏览器麦克风直接录音
- **扫描 input/ 文件夹**：批量读取 `input/` 目录下的所有音频

**时频特征展示**：
- 时域波形图：横轴时间(秒)，纵轴振幅。波形越密集 = 频率越高。
- 频谱图：对数幅度谱，展示频率成分分布，谐波峰间距对应基频 F0。
- 梅尔频谱图：80 通道梅尔频谱 (dB)，横轴时间帧，纵轴梅尔频率。可观察共振峰 F1/F2/F3 轨迹，是区分 TTS 与真声的关键依据。
- 8 个时域特征 + 10 个频域特征的数值指标卡（悬停可见中文解释）。

**识别分析**（点击"执行识别与分析"按钮）：
- 说话人识别结果 + 声纹余弦相似度
- TTS 检测：分数 + 真/伪判定
- 展开"TTS 检测明细"可查看 3 个维度的子分数（底噪稳定性、高频平坦度波动、底噪强度）
- ASR 语音识别转写文本
- JSON 完整结果输出 + 下载按钮

### 标签页二：说话人注册

将每个说话人的 3~5 条音频按如下结构放入 `register_id/`：

```
register_id/
├── 张三/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── sample_003.wav
└── 李四/
    ├── audio_01.wav
    └── audio_02.wav
```

点击"注册/更新声纹库"按钮即可。注册信息保存在 `data/speaker_db.json`。

> 建议录音环境安静，采样率推荐 16kHz，每条音频 3~10 秒。

## 技术架构

### 算法链路

```
音频输入 (16kHz Mono)
    │
    ├── 时频特征提取 ──→ 18 个特征 + 波形图/频谱图/梅尔频谱图
    │
    ├── ECAPA-TDNN 嵌入 (192维)
    │   └── Cosine 相似度匹配 → 说话人识别结果
    │
    ├── TTS 底噪检测 (三维评分)
    │   ├── 底噪稳定性 (权重 0.50) — 静音段 RMS 变异系数
    │   ├── 高频平坦度波动 (权重 0.30) — 3kHz+ 平坦度帧间标准差
    │   └── 底噪强度 (权重 0.20) — 静音段 RMS 量级
    │
    └── Paraformer ASR → 中文转写文本
```

### 核心模型

| 模块 | 模型 | 参数 | 来源 |
|------|------|------|------|
| 声纹识别 | ECAPA-TDNN | 嵌入维度 192 | SpeechBrain / VoxCeleb |
| 语音识别 | Paraformer-large | 非自回归 Transformer | FunASR / DAMO |
| TTS 检测 | 自定义启发式 | 3 维加权评分 | 底噪分析 |

### 可调参数

| 参数 | 默认值 | 范围 | 说明 |
|------|:--:|------|------|
| 说话人相似度阈值 | 0.62 | 0.40~0.95 | 声纹匹配判定阈值 |
| TTS 检测阈值 | 0.40 | 0.10~0.95 | TTS 综合评分阈值，≥ 此值为真人 |

### TTS 检测原理

真实语音经过麦克风 → ADC 采集链路，即使"安静"环境下也存在底噪（RMS ≈ 10⁻⁵ ~ 10⁻³），且底噪有自然时变波动（CV > 0.4）。

TTS 合成语音全程在数字域生成，不经过模拟器件，静音段呈现"数字死寂"特征：
- 底噪异常恒定（CV < 0.15）
- 高频段频谱过度平滑（平坦度标准差 < 0.07）

## 输出格式

单文件分析输出 JSON 示例：

```json
{
  "speaker": "张三",
  "similarity": 0.8523,
  "is_genuine": true,
  "tts_score": 0.8734,
  "tts_details": {
    "noise_floor_rms": 0.00215,
    "noise_cv": 0.523,
    "noise_stability_score": 0.946,
    "hf_flat_std": 0.1601,
    "hf_var_score": 1.0
  },
  "asr_paraformer": "今天天气很好适合出门散步",
  "time_domain_features": { "duration_sec": 3.2, "rms_mean": 0.0495, ... },
  "frequency_domain_features": { "spectral_centroid_mean": 1930.3, ... },
  "speaker_similarities": { "张三": 0.8523, "李四": 0.3124 }
}
```

## 常见问题

**Q: 启动后报 `ModuleNotFoundError`**  
A: 确保已激活虚拟环境并执行 `pip install -r requirements.txt`。

**Q: 声纹模型加载失败**  
A: 检查网络连接，确保可访问 HuggingFace。或手动下载模型放到 `models/speaker/ecapa/`。

**Q: 注册后识别时提示"声纹库为空"**  
A: 检查 `register_id/` 下是否按说话人姓名建了子文件夹，子文件夹内是否有音频文件。

**Q: TTS 分数对真人和 TTS 区分不明显**  
A: 调低 TTS 检测阈值（侧边栏滑块），或在安静环境下重新录制注册音频以提高底噪特征质量。

**Q: CPU 内存不足**  
A: Paraformer 模型较大（~848MB），确保至少 4GB 可用内存。如需进一步精简可移除 ASR 模块。
