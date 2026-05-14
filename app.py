from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import librosa
import pandas as pd
import streamlit as st

from signal_system import (
    VoiceAntiSpoofSystem,
    extract_time_features,
    extract_freq_features,
    plot_time_waveform,
    plot_frequency_spectrum,
    plot_mel_spectrogram,
    plot_features_dashboard,
    SUPPORTED_AUDIO,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SPK_MODEL = ROOT / "models" / "speaker" / "ecapa"
DEFAULT_ASR_MODEL = ROOT / "models" / "asr" / "damo" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_ENROLL_DIR = ROOT / "register_id"
DEFAULT_DB = ROOT / "data" / "speaker_db.json"
INPUT_DIR = ROOT / "input"

# 确保文件夹存在
INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_ENROLL_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def build_system(spk_model: str, asr_model: str, db_path: str, sim_thr: float, prof_thr: float):
    system = VoiceAntiSpoofSystem(
        speaker_model_dir=Path(spk_model),
        paraformer_path=Path(asr_model),
        db_path=Path(db_path),
        speaker_threshold=sim_thr,
        profile_threshold=prof_thr,
    )
    system.load_models()
    return system


def read_uploaded_audio(uploaded_file) -> tuple:
    # 先读取全部字节，避免重复 .read() 返回空数据
    if hasattr(uploaded_file, "getvalue"):       # BytesIO (st.audio_input)
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):         # UploadedFile
        data = uploaded_file.read()
    else:
        data = uploaded_file
    if not data:
        raise ValueError("音频数据为空，请重新选择文件或录音。")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        y, sr = librosa.load(str(tmp_path), sr=16000, mono=True)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return y, sr


# ── 页面配置 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="语音信号分析系统",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("语音信号分析系统")
st.caption("音频输入 · 说话人注册 · 时频特征提取 · ASR识别 · JSON输出")

# ── 侧边栏配置 ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("系统配置")
    spk_model_dir = st.text_input("声纹模型目录", str(DEFAULT_SPK_MODEL))
    asr_model_dir = st.text_input("ASR模型目录", str(DEFAULT_ASR_MODEL))
    db_path = st.text_input("声纹库文件", str(DEFAULT_DB))
    enroll_dir = st.text_input("注册语音目录 (register_id/)", str(DEFAULT_ENROLL_DIR))
    sim_thr = st.slider("说话人相似度阈值", 0.40, 0.95, 0.62, 0.01)
    prof_thr = st.slider("TTS检测阈值", 0.10, 0.95, 0.40, 0.01, help="综合TTS分数阈值，≥该值判定为真人（非TTS合成）")

    st.divider()
    st.markdown("### 文件夹结构说明")
    st.code("""
signal/
├── input/          ← 放入待分析音频
├── register_id/    ← 按说话人建立子文件夹
│   ├── speaker_1/  ← 每人3-5条音频
│   ├── speaker_2/
│   └── ...
└── data/
    └── speaker_db.json
    """)

system = build_system(spk_model_dir, asr_model_dir, db_path, sim_thr, prof_thr)

# ── 标签页 ────────────────────────────────────────────────────────────
tab_input, tab_register = st.tabs([
    "音频输入与分析",
    "说话人注册",
])

# ═════════════════════════════════════════════════════════════════════
# Tab 1: 音频输入与分析
# ═════════════════════════════════════════════════════════════════════
with tab_input:
    col_left, col_right = st.columns([1, 1])

    # ── 左侧：音频输入 ───────────────────────────────────────────────
    with col_left:
        st.subheader("音频输入")
        input_mode = st.radio("选择输入方式", ["上传文件", "现场录音", "扫描 input/ 文件夹"],
                              horizontal=True)

        audio_obj = None
        input_files = []

        if input_mode == "上传文件":
            uploaded = st.file_uploader(
                "选择音频文件", type=["wav", "mp3", "flac", "m4a", "ogg"],
                accept_multiple_files=False,
            )
            if uploaded:
                audio_obj = uploaded
                st.audio(uploaded)
        elif input_mode == "现场录音":
            mic_audio = st.audio_input("点击录音")
            if mic_audio:
                audio_obj = mic_audio
        else:
            # 扫描 input/ 文件夹
            if INPUT_DIR.exists():
                audio_files = sorted([
                    f for f in INPUT_DIR.rglob("*")
                    if f.suffix.lower() in SUPPORTED_AUDIO
                ])
                input_files = audio_files
                if audio_files:
                    st.info(f"发现 **{len(audio_files)}** 个音频文件")
                    file_names = [f"{f.name} ({f.relative_to(INPUT_DIR)})" for f in audio_files]
                    st.dataframe(pd.DataFrame({"文件": file_names}), use_container_width=True)
                else:
                    st.warning("`input/` 文件夹为空，请放入音频文件。")

        if audio_obj is not None:
            y, sr = read_uploaded_audio(audio_obj)
            st.write(f"采样率：{sr} Hz | 时长：{len(y) / sr:.2f} s | 样本数：{len(y)}")
        else:
            y, sr = None, None

    # ── 右侧：快速分析 ───────────────────────────────────────────────
    with col_right:
        st.subheader("时频特征分析")

        # 单文件分析（复用左侧已读取的 y, sr）
        if y is not None and sr is not None:

            with st.expander("时域波形图", expanded=True):
                fig_t = plot_time_waveform(y, sr)
                st.pyplot(fig_t, clear_figure=True)
                st.caption("横轴为时间(秒)，纵轴为采样振幅。波形越密集=频率越高，振幅越大=响度越大。真实语音有自然的起止和停顿。")

            with st.expander("频谱图", expanded=True):
                fig_f = plot_frequency_spectrum(y, sr)
                st.pyplot(fig_f, clear_figure=True)
                st.caption("横轴为频率(Hz)，纵轴为幅度(对数刻度)。展示信号的频率成分分布，谐波峰之间的间距对应基频F0。")

            with st.expander("梅尔频谱图", expanded=True):
                fig_m = plot_mel_spectrogram(y, sr)
                st.pyplot(fig_m, clear_figure=True)
                st.caption("横轴为时间帧，纵轴为梅尔频率(模拟人耳感知)，颜色深度代表能量(dB)。能看到共振峰(F1/F2/F3)随时间的变化轨迹，是区分TTS与真声的关键依据。")

            # 提取特征
            t_feat = extract_time_features(y, sr)
            f_feat = extract_freq_features(y, sr)

            st.markdown("**时域特征 (Time Domain)**")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("时长 (s)", t_feat.duration_sec, help="音频信号的总时间长度")
                st.metric("RMS 均值", f"{t_feat.rms_mean:.6f}", help="均方根能量均值，反映整体响度大小")
                st.metric("ZCR 均值", f"{t_feat.zcr_mean:.6f}", help="过零率均值，信号穿过零轴的频率，反映高频成分占比")
                st.metric("最大振幅", f"{t_feat.max_amplitude:.6f}", help="信号采样点的最大绝对幅度")
            with col_t2:
                st.metric("RMS 标准差", f"{t_feat.rms_std:.6f}", help="均方根能量的波动程度，反映音量稳定性")
                st.metric("ZCR 标准差", f"{t_feat.zcr_std:.6f}", help="过零率随时间的变化程度")
                st.metric("波峰因子", f"{t_feat.crest_factor:.2f}", help="峰值与RMS之比，衡量信号的冲击性和动态范围")
                st.metric("峰峰值", f"{t_feat.peak_to_peak:.6f}", help="信号正负最大幅度的差值，反映最大摆幅")

            st.markdown("**频域特征 (Frequency Domain)**")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.metric("频谱质心均值 (Hz)", f"{f_feat.spectral_centroid_mean:.1f}", help="频谱的「重心」频率，越高音色越明亮/尖锐")
                st.metric("频谱带宽均值 (Hz)", f"{f_feat.spectral_bandwidth_mean:.1f}", help="频谱能量分布的宽度，反映音色复杂度")
                st.metric("频谱平坦度均值", f"{f_feat.spectral_flatness_mean:.6f}", help="频谱平坦程度，越接近0越有调性(谐波结构明显)，越接近1越像噪声")
                st.metric("HF能量比", f"{f_feat.hf_energy_ratio:.4f}", help="4kHz以上高频能量占总能量的比例，录音重放通常偏低")
            with col_f2:
                st.metric("频谱质心标准差", f"{f_feat.spectral_centroid_std:.1f}", help="频谱质心随时间的变化程度，反映音色稳定性")
                st.metric("频谱带宽标准差", f"{f_feat.spectral_bandwidth_std:.1f}", help="频谱带宽随时间的变化程度")
                st.metric("频谱滚降均值 (Hz)", f"{f_feat.spectral_rolloff_mean:.1f}", help="累积频谱能量达到85%时的频率，反映频谱能量的集中度")
                st.metric("频谱熵", f"{f_feat.spectral_entropy:.4f}", help="频谱的信息熵，值越高频谱结构越复杂无序")

    # ── 底部：识别 + JSON 输出 ──────────────────────────────────────
    if audio_obj is not None:
        st.divider()
        st.subheader("识别分析与 JSON 输出")

        if st.button("执行识别与分析", type="primary", use_container_width=True):
            # 重新读取一份（uploaded_file 可能已被消费，这里独立读取）
            try:
                _y, _sr = read_uploaded_audio(audio_obj)
            except (ValueError, EOFError) as e:
                st.error(f"读取音频失败：{e}")
                st.stop()
            try:
                result = system.recognize_and_detect(_y, _sr)
            except RuntimeError:
                st.error("声纹库为空，请先在「说话人注册」标签页注册说话人。")
                st.stop()

            # 说话人识别
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("识别说话人", result.speaker)
            with col_r2:
                st.metric("相似度", f"{result.similarity:.4f}", help="声纹余弦相似度，阈值≥0.62为匹配")
            with col_r3:
                tts_pass = result.tts_score >= prof_thr
                tts_text = "真声 ✓" if tts_pass else "TTS合成 ✗"
                st.metric("TTS分数", f"{result.tts_score:.4f}", help="综合评分：1.0=真人，0.0=TTS合成")
            with col_r4:
                st.metric("TTS检测", tts_text)

            # TTS 子分数明细
            if result.tts_details:
                with st.expander("TTS检测明细"):
                    d = result.tts_details
                    st.markdown("**底噪稳定性** (权重0.50 — 核心区分维度)")
                    nc1, nc2, nc3 = st.columns(3)
                    with nc1:
                        st.metric("底噪CV", f"{d['noise_cv']:.3f}", help="底噪变异系数，真人>0.4(自然波动)，TTS<0.15(异常恒定)")
                    with nc2:
                        st.metric("稳定性分数", f"{d['noise_stability_score']:.3f}")
                    with nc3:
                        st.metric("底噪RMS", f"{d['noise_floor_rms']:.2e}", help="安静段平均RMS能量")

                    st.markdown("**高频平坦度波动** (权重0.30)")
                    hc1, hc2, hc3 = st.columns(3)
                    with hc1:
                        st.metric("平坦度Std", f"{d['hf_flat_std']:.4f}", help="高频平坦度帧间标准差，真人>0.12(谐波自然波动)，TTS<0.07(均匀平滑)")
                    with hc2:
                        st.metric("波动分数", f"{d['hf_var_score']:.3f}")
                    with hc3:
                        st.metric("平坦度均值", f"{d['hf_flatness_3kHz']:.4f}", help="仅参考，均值差异小")

                    st.markdown("**底噪强度** (权重0.20 — 辅助)")
                    st.metric("底噪强度分数", f"{d['noise_floor_score']:.3f}", help="底噪RMS评分：死寂(<1e-6)→0，有底噪(>5e-4)→0.85+")

            # 所有说话人相似度
            if result.speaker_similarities:
                st.markdown("**说话人相似度：**")
                sim_df = pd.DataFrame(
                    {"说话人": list(result.speaker_similarities.keys()),
                     "相似度": list(result.speaker_similarities.values())}
                ).sort_values("相似度", ascending=False)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

            # ASR 结果
            st.markdown("**语音识别结果 (ASR)：**")
            st.text_area("Paraformer", result.paraformer_text or "(空)", height=80)

            # JSON 输出
            st.markdown("**JSON 输出：**")
            json_str = result.to_json(indent=2)
            st.json(json_str)
            st.download_button(
                "下载 JSON 结果",
                json_str,
                file_name=f"analysis_{result.speaker}.json",
                mime="application/json",
            )

    # ── 批量处理 input/ 文件夹 ─────────────────────────────────────
    if input_mode == "扫描 input/ 文件夹" and input_files:
        st.divider()
        st.subheader("批量分析 input/ 文件夹")
        if st.button("批量分析并导出 JSON", type="primary", use_container_width=True):
            with st.spinner(f"正在分析 {len(input_files)} 个文件..."):
                try:
                    results = system.process_input_folder(INPUT_DIR)
                except RuntimeError:
                    st.error("声纹库为空，请先在「说话人注册」标签页注册说话人。")
                    st.stop()

            success_count = sum(1 for r in results if "error" not in r)
            error_count = len(results) - success_count
            st.success(f"完成：{success_count} 成功, {error_count} 失败")

            output_json = json.dumps(results, ensure_ascii=False, indent=2)
            st.json(output_json)
            st.download_button(
                "下载批量分析 JSON",
                output_json,
                file_name="batch_analysis_results.json",
                mime="application/json",
            )

            # 汇总表
            if success_count > 0:
                summary_rows = []
                for r in results:
                    if "error" not in r:
                        summary_rows.append({
                            "文件": r.get("file", ""),
                            "说话人": r.get("speaker", ""),
                            "相似度": r.get("similarity", 0),
                            "真伪": "真" if r.get("is_genuine") else "伪",
                            "时长(s)": r.get("duration_sec", 0),
                            "ASR结果": r.get("asr_paraformer", "")[:50],
                        })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════
# Tab 2: 说话人注册
# ═════════════════════════════════════════════════════════════════════
with tab_register:
    st.subheader("说话人注册管理")

    col_reg1, col_reg2 = st.columns([2, 1])

    with col_reg1:
        st.markdown("""
        ### 注册方式
        将每个说话人的音频放入 `register_id/` 下独立的子文件夹：

        ```
        register_id/
        ├── zhangsan/
        │   ├── sample_001.wav
        │   ├── sample_002.wav
        │   └── sample_003.wav
        ├── lisi/
        │   ├── audio_01.wav
        │   └── audio_02.wav
        └── ...
        ```

        每位说话人建议 **3~5 条** 清晰音频，格式支持 WAV / MP3 / FLAC / M4A / OGG。
        """)

    with col_reg2:
        # 显示当前注册状态
        db_file = Path(db_path)
        if db_file.exists():
            raw = json.loads(db_file.read_text(encoding="utf-8"))
            st.info(f"已注册 **{len(raw)}** 位说话人")
            st.write(", ".join(raw.keys()))
        else:
            st.warning("尚未注册说话人")

        # 查看 register_id 文件夹内容
        reg_dir = Path(enroll_dir)
        if reg_dir.exists():
            reg_folders = [d for d in reg_dir.iterdir() if d.is_dir()]
            if reg_folders:
                with st.expander("register_id/ 文件夹内容"):
                    for d in reg_folders:
                        files = list(d.rglob("*"))
                        audio_count = sum(1 for f in files if f.suffix.lower() in SUPPORTED_AUDIO)
                        st.write(f"- `{d.name}/`  ({audio_count} 条音频)")

    # 注册按钮
    if st.button("注册/更新声纹库", type="primary", use_container_width=True):
        with st.spinner("正在注册..."):
            summary = system.enroll_from_directory(Path(enroll_dir))
        if not summary:
            st.error("未找到可用注册音频，请检查 `register_id/` 目录。")
        else:
            st.success(f"注册完成！")
            for name, count in summary.items():
                st.write(f"- `{name}`: {count} 条音频")

            # 显示注册后的时频特征
            if system.speaker_db:
                st.markdown("### 注册说话人特征统计")
                for name, info in system.speaker_db.items():
                    with st.expander(f"{name} 防伪特征"):
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown("**均值 (Profile Mean)**")
                            st.json(info["profile_mean"])
                        with col_s2:
                            st.markdown("**标准差 (Profile Std)**")
                            st.json(info["profile_std"])
