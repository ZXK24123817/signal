# 语音信号分析系统

基于 Streamlit 的语音信号分析 Web 应用，集成**说话人声纹识别**、**TTS 防伪检测**、**中文语音识别 (ASR)** 和**时频域特征提取**。

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动
streamlit run app.py
```

首次运行会自动从 HuggingFace / ModelScope 下载预训练模型（约 1GB），请确保网络畅通。

## 功能

| 模块 | 功能 | 技术方案 |
|------|------|----------|
| 音频输入 | 文件上传 / 麦克风录音 / 文件夹批量扫描 | Streamlit |
| 时频分析 | 8 时域 + 10 频域特征，波形图/频谱图/梅尔频谱图 | Librosa + Matplotlib |
| 声纹识别 | 说话人注册与识别，余弦相似度匹配 | ECAPA-TDNN (SpeechBrain) |
| TTS 检测 | 区分真声与合成语音，三维加权评分 | 底噪稳定性分析 |
| 语音识别 | 中文语音转文字 | Paraformer (FunASR) |
| 结果输出 | 结构化 JSON + 下载 | — |

## 目录结构

```
signal/
├── app.py              # Streamlit 界面
├── signal_system.py    # 核心算法
├── requirements.txt    # Python 依赖
├── start.md            # 详细文档
├── register_id/        # 说话人注册音频
├── input/              # 待批量分析音频（可选）
└── data/               # 声纹库 JSON（运行时生成）
```

## 可调参数

| 参数 | 默认值 | 说明 |
|------|:--:|------|
| 声纹相似度阈值 | 0.62 | 高于此值判定为同一说话人 |
| TTS 检测阈值 | 0.40 | 高于此值判定为真声 |

## 详细文档

参见 [start.md](start.md)
