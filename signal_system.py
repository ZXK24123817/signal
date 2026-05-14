from __future__ import annotations

import json
import tempfile
import types
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _init_speechbrain_lazy_patch():
    """
    SpeechBrain 的 LazyModule 在 Python 3.13 下与 inspect.stack() 冲突：
    hasattr(module, '__file__') 会触发 k2_fsa/nlp 等未安装模块的懒加载并抛
    ImportError。这里将 ensure_module 包装为失败时返回空模块。
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except ImportError:
        return

    _fail_set = set()
    _orig = LazyModule.ensure_module

    def _safe(self, stacklevel=1):
        key = self.target
        if key in _fail_set:
            return types.ModuleType(key)
        try:
            return _orig(self, stacklevel + 1)
        except ImportError:
            _fail_set.add(key)
            return types.ModuleType(key)

    LazyModule.ensure_module = _safe


_init_speechbrain_lazy_patch()


import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch


SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# ── Matplotlib 中文字体设置 ──────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TimeDomainFeatures:
    duration_sec: float
    rms_mean: float
    rms_std: float
    zcr_mean: float
    zcr_std: float
    max_amplitude: float
    peak_to_peak: float
    crest_factor: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrequencyDomainFeatures:
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_std: float
    spectral_flatness_mean: float
    spectral_flatness_std: float
    spectral_rolloff_mean: float
    spectral_rolloff_std: float
    hf_energy_ratio: float
    spectral_entropy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecognitionResult:
    speaker: str
    similarity: float
    is_genuine: bool
    tts_score: float
    tts_details: Optional[dict] = None
    paraformer_text: str = ""
    time_features: Optional[dict] = None
    freq_features: Optional[dict] = None
    speaker_similarities: Optional[dict] = None

    def to_json(self, indent: int = 2) -> str:
        d = {
            "speaker": self.speaker,
            "similarity": round(self.similarity, 4),
            "is_genuine": self.is_genuine,
            "tts_score": round(self.tts_score, 4),
            "tts_details": self.tts_details or {},
            "asr_paraformer": self.paraformer_text or "",
            "time_domain_features": self.time_features or {},
            "frequency_domain_features": self.freq_features or {},
        }
        if self.speaker_similarities:
            d["speaker_similarities"] = {k: round(v, 4) for k, v in self.speaker_similarities.items()}
        return json.dumps(d, ensure_ascii=False, indent=indent)


# ═══════════════════════════════════════════════════════════════════════
# 语音识别后端
# ═══════════════════════════════════════════════════════════════════════

class ASRBackends:
    def __init__(self, paraformer_path: Path):
        self._paraformer = None
        self._paraformer_path = paraformer_path

    def _load_paraformer(self) -> None:
        if self._paraformer is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError:
            return
        import torch

        torch.set_num_threads(2)

        local_path = str(self._paraformer_path)
        model_pt = self._paraformer_path / "model.pt"
        if model_pt.exists() and model_pt.stat().st_size > 0:
            try:
                self._paraformer = AutoModel(
                    model=local_path, device="cpu",
                    disable_update=True, ncpu=2,
                )
                return
            except Exception:
                pass

        model_name = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        try:
            self._paraformer = AutoModel(
                model=model_name, device="cpu",
                disable_update=True, ncpu=2,
            )
        except Exception:
            self._paraformer = None

    def transcribe(self, wav_path: Path) -> str:
        return self._transcribe_paraformer(wav_path)

    def _transcribe_paraformer(self, wav_path: Path) -> str:
        self._load_paraformer()
        if self._paraformer is None:
            return ""
        try:
            output = self._paraformer.generate(input=str(wav_path))
            if not output:
                return ""
            first = output[0]
            if isinstance(first, dict):
                return str(first.get("text", "")).strip()
            return str(first).strip()
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════════════
# 特征提取
# ═══════════════════════════════════════════════════════════════════════

def extract_time_features(y: np.ndarray, sr: int) -> TimeDomainFeatures:
    """提取全面的时域特征"""
    duration = len(y) / sr
    frame_len = 1024
    hop_len = 256

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    max_amp = float(np.max(np.abs(y)))
    peak_to_peak = float(np.max(y) - np.min(y))

    crest = max_amp / (rms_mean + 1e-8)

    return TimeDomainFeatures(
        duration_sec=round(duration, 3),
        rms_mean=round(rms_mean, 6),
        rms_std=round(rms_std, 6),
        zcr_mean=round(zcr_mean, 6),
        zcr_std=round(zcr_std, 6),
        max_amplitude=round(max_amp, 6),
        peak_to_peak=round(peak_to_peak, 6),
        crest_factor=round(float(crest), 4),
    )


def extract_freq_features(y: np.ndarray, sr: int) -> FrequencyDomainFeatures:
    """提取全面的频域特征"""
    n_fft = 1024
    hop_len = 256

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_len)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_len)[0]
    bandwidth_mean = float(np.mean(bandwidth))
    bandwidth_std = float(np.std(bandwidth))

    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_len)[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    hf_mask = freqs >= 4000
    spec_energy = stft.mean(axis=1) + 1e-8
    hf_ratio = float(spec_energy[hf_mask].sum() / spec_energy.sum())

    spec_norm = stft / (stft.sum(axis=0, keepdims=True) + 1e-8)
    entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-12), axis=0)
    spectral_entropy = float(np.mean(entropy))

    return FrequencyDomainFeatures(
        spectral_centroid_mean=round(centroid_mean, 2),
        spectral_centroid_std=round(centroid_std, 2),
        spectral_bandwidth_mean=round(bandwidth_mean, 2),
        spectral_bandwidth_std=round(bandwidth_std, 2),
        spectral_flatness_mean=round(flatness_mean, 6),
        spectral_flatness_std=round(flatness_std, 6),
        spectral_rolloff_mean=round(rolloff_mean, 2),
        spectral_rolloff_std=round(rolloff_std, 2),
        hf_energy_ratio=round(hf_ratio, 4),
        spectral_entropy=round(spectral_entropy, 4),
    )


# ═══════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════

def plot_time_waveform(y: np.ndarray, sr: int) -> plt.Figure:
    """时域波形图"""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    t = np.arange(len(y)) / sr
    ax.plot(t, y, linewidth=0.6, color="#1f77b4")
    ax.set_title("时域波形 (Time Domain Waveform)", fontsize=13)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("振幅")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_frequency_spectrum(y: np.ndarray, sr: int) -> plt.Figure:
    """频谱图"""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    n_fft = 4096
    yf = np.fft.rfft(y[:n_fft] * np.hanning(min(n_fft, len(y))))
    xf = np.fft.rfftfreq(min(n_fft, len(y)), 1 / sr)
    ax.semilogy(xf, np.abs(yf), linewidth=0.6, color="#ff7f0e")
    ax.set_title("频谱图 (Frequency Spectrum)", fontsize=13)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度 (log)")
    ax.set_xlim(0, sr / 2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(y: np.ndarray, sr: int) -> plt.Figure:
    """梅尔频谱图"""
    fig, ax = plt.subplots(figsize=(10, 3))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("梅尔频谱图 (Mel Spectrogram)", fontsize=13)
    ax.set_xlabel("帧 (Frame)")
    ax.set_ylabel("梅尔频率 (Mel Bin)")
    fig.colorbar(img, ax=ax, fraction=0.02, pad=0.01, label="dB")
    fig.tight_layout()
    return fig


def plot_features_dashboard(y: np.ndarray, sr: int) -> plt.Figure:
    """综合展示：时域波形 + 频谱 + 梅尔频谱"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # 时域波形
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, linewidth=0.6, color="#1f77b4")
    axes[0].set_title("时域波形 (Time Domain)", fontsize=13)
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("振幅")
    axes[0].grid(True, alpha=0.3)

    # 频谱
    n_fft = 4096
    n = min(n_fft, len(y))
    yf = np.fft.rfft(y[:n] * np.hanning(n))
    xf = np.fft.rfftfreq(n, 1 / sr)
    axes[1].semilogy(xf, np.abs(yf), linewidth=0.6, color="#ff7f0e")
    axes[1].set_title("频谱 (Frequency Spectrum)", fontsize=13)
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("幅度 (log)")
    axes[1].set_xlim(0, sr / 2)
    axes[1].grid(True, alpha=0.3)

    # 梅尔频谱
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = axes[2].imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    axes[2].set_title("梅尔频谱 (Mel Spectrogram)", fontsize=13)
    axes[2].set_xlabel("帧 (Frame)")
    axes[2].set_ylabel("梅尔频率 (Mel Bin)")
    fig.colorbar(img, ax=axes[2], fraction=0.02, pad=0.01, label="dB")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 主系统
# ═══════════════════════════════════════════════════════════════════════

class VoiceAntiSpoofSystem:
    def __init__(
        self,
        speaker_model_dir: Path,
        paraformer_path: Path,
        db_path: Path,
        speaker_threshold: float = 0.62,
        profile_threshold: float = 0.45,
    ) -> None:
        self.speaker_model_dir = speaker_model_dir
        self.db_path = db_path
        self.speaker_threshold = speaker_threshold
        self.profile_threshold = profile_threshold
        self.speaker_model = None
        self.speaker_db: Dict[str, Dict] = {}
        self.asr = ASRBackends(paraformer_path)

    # ── 模型加载 ─────────────────────────────────────────────────────

    def load_models(self) -> None:
        if self.speaker_model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self.speaker_model = EncoderClassifier.from_hparams(
                source=str(self.speaker_model_dir),
                run_opts={"device": "cpu"},
            )
        self._load_db()

    def _load_db(self) -> None:
        if not self.db_path.exists():
            self.speaker_db = {}
            return
        raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        db: Dict[str, Dict] = {}
        for name, values in raw.items():
            db[name] = {
                "centroid": np.asarray(values["centroid"], dtype=np.float32),
                "profile_mean": values["profile_mean"],
                "profile_std": values["profile_std"],
            }
        self.speaker_db = db

    def _save_db(self) -> None:
        data = {}
        for speaker, values in self.speaker_db.items():
            data[speaker] = {
                "centroid": values["centroid"].tolist(),
                "profile_mean": values["profile_mean"],
                "profile_std": values["profile_std"],
            }
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 工具方法 ─────────────────────────────────────────────────────

    @staticmethod
    def load_audio(audio_path: Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        return y.astype(np.float32), sr

    @staticmethod
    def signal_profile(y: np.ndarray, sr: int) -> Dict[str, float]:
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        hf_mask = freqs >= 4000
        spec_energy = stft.mean(axis=1) + 1e-8
        hf_ratio = float(spec_energy[hf_mask].sum() / spec_energy.sum())
        flatness = float(librosa.feature.spectral_flatness(S=stft).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).mean())
        rms = float(librosa.feature.rms(y=y, frame_length=1024, hop_length=256).mean())

        # 频谱熵：反映频谱结构复杂度，TTS通常偏低
        spec_norm = stft / (stft.sum(axis=0, keepdims=True) + 1e-8)
        entropy_frames = -np.sum(spec_norm * np.log(spec_norm + 1e-8), axis=0)
        spec_entropy = float(np.mean(entropy_frames))

        return {"hf_ratio": hf_ratio, "flatness": flatness, "zcr": zcr, "rms": rms, "spec_entropy": spec_entropy}

    def embedding(self, y: np.ndarray) -> np.ndarray:
        if self.speaker_model is None:
            raise RuntimeError("Speaker model not loaded.")
        tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.speaker_model.encode_batch(tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb) + 1e-8
        return (emb / norm).astype(np.float32)

    # ── 注册 ─────────────────────────────────────────────────────────

    def enroll_from_directory(self, enroll_root: Path) -> Dict[str, int]:
        self.load_models()
        summary: Dict[str, int] = {}
        db: Dict[str, Dict] = {}
        for speaker_dir in sorted(p for p in enroll_root.iterdir() if p.is_dir()):
            files = [f for f in speaker_dir.rglob("*") if f.suffix.lower() in SUPPORTED_AUDIO]
            if not files:
                continue
            embs = []
            profiles = []
            for file in files:
                y, sr = self.load_audio(file)
                if len(y) < sr:
                    continue
                embs.append(self.embedding(y))
                profiles.append(self.signal_profile(y, sr))
            if not embs:
                continue
            centroid = np.mean(np.stack(embs), axis=0).astype(np.float32)
            centroid /= np.linalg.norm(centroid) + 1e-8
            keys = profiles[0].keys()
            profile_mean = {k: float(np.mean([p[k] for p in profiles])) for k in keys}
            profile_std = {}
            for k in keys:
                raw_std = float(np.std([p[k] for p in profiles]))
                floor = max(abs(profile_mean[k]) * 0.05, 0.005)
                profile_std[k] = max(raw_std, floor)
            db[speaker_dir.name] = {
                "centroid": centroid,
                "profile_mean": profile_mean,
                "profile_std": profile_std,
            }
            summary[speaker_dir.name] = len(embs)
        self.speaker_db = db
        self._save_db()
        return summary

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

    def _tts_score(self, y: np.ndarray, sr: int) -> Tuple[float, Dict[str, float]]:
        """TTS检测: 返回 (总分, 子分数明细).

        经实测数据校准，有效区分维度：
        1. 底噪稳定性 (0.50) — TTS底噪异常恒定(cv<0.15)，真人自然波动(cv>0.4)
        2. 高频平坦度波动 (0.30) — 真人谐波结构随帧变化，TTS均匀平滑
        3. 底噪强度 (0.20) — 辅助判断，死寂(<1e-6)标志TTS
        """
        frame_len = 2048
        hop = 512
        n_fft = 2048

        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop).squeeze()
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))

        # ═══ 1. 底噪稳定性 (核心区分维度) ═══
        quiet_floor = np.percentile(rms, 15)
        quiet_mask = rms <= quiet_floor
        quiet_rms = rms[quiet_mask] if np.any(quiet_mask) else rms[:1]
        noise_floor = float(np.mean(quiet_rms))
        noise_cv = float(np.std(quiet_rms) / (np.mean(quiet_rms) + 1e-8))

        # 底噪强度分: 死寂→0, 微弱→0.25, 合理→0.7+, 明显→0.9
        if noise_floor < 1e-6:
            nf = 0.0
        elif noise_floor < 1e-5:
            nf = 0.25
        elif noise_floor < 5e-4:
            nf = 0.65 + 0.1 * min(1.0, noise_floor / 5e-4)
        else:
            nf = 0.85

        # 底噪稳定性分: cv>0.5→真人(1.0), cv<0.1→TTS(0.1), 线性映射
        ns = float(np.clip((noise_cv - 0.05) / 0.50, 0.0, 1.0))
        noise_stability_score = ns
        noise_floor_score = nf
        noise_score = float(0.5 * nf + 0.5 * ns)

        # ═══ 2. 高频平坦度帧间波动 (区分维度) ═══
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hf_mask = freqs >= 3000
        if hf_mask.any():
            hf_flat_frames = librosa.feature.spectral_flatness(S=stft[hf_mask, :]).squeeze()
            hf_flat_mean = float(np.mean(hf_flat_frames))
            hf_flat_std = float(np.std(hf_flat_frames))
        else:
            hf_flat_mean = 0.0
            hf_flat_std = 0.0

        # 平坦度波动: std>0.12→真人(1.0), std<0.05→TTS, 真人帧间谐波有起伏
        hf_var_score = float(np.clip(hf_flat_std / 0.12, 0.0, 1.0))
        # 均值仅供参考，权重低
        hf_mean_score = float(np.clip(1.0 - hf_flat_mean / 0.20, 0.0, 1.0))
        hf_score = float(0.75 * hf_var_score + 0.25 * hf_mean_score)

        # ═══ 3. 加权综合 ═══
        total = float(0.50 * noise_stability_score + 0.30 * hf_score + 0.20 * noise_floor_score)

        details = {
            "noise_floor_rms": round(noise_floor, 8),
            "noise_cv": round(noise_cv, 3),
            "noise_stability_score": round(noise_stability_score, 3),
            "noise_floor_score": round(noise_floor_score, 3),
            "noise_combined": round(noise_score, 3),
            "hf_flatness_3kHz": round(hf_flat_mean, 4),
            "hf_flat_std": round(hf_flat_std, 4),
            "hf_var_score": round(hf_var_score, 3),
            "hf_mean_score": round(hf_mean_score, 3),
            "hf_combined": round(hf_score, 3),
        }
        return total, details

    # ── 识别 + 检测 ─────────────────────────────────────────────────

    def recognize_and_detect(self, y: np.ndarray, sr: int) -> RecognitionResult:
        if not self.speaker_db:
            raise RuntimeError("Speaker DB is empty. Please enroll first.")

        emb = self.embedding(y)
        sims = {name: self._cosine(emb, values["centroid"]) for name, values in self.speaker_db.items()}
        speaker = max(sims, key=sims.get)
        similarity = float(sims[speaker])

        tts_score, tts_details = self._tts_score(y, sr)

        is_genuine = similarity >= self.speaker_threshold and tts_score >= self.profile_threshold

        time_feat = extract_time_features(y, sr)
        freq_feat = extract_freq_features(y, sr)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(tmp_path, y, sr)
        paraformer_text = self.asr.transcribe(tmp_path)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        return RecognitionResult(
            speaker=speaker,
            similarity=similarity,
            is_genuine=is_genuine,
            tts_score=tts_score,
            tts_details=tts_details,
            paraformer_text=paraformer_text or "",
            time_features=time_feat.to_dict(),
            freq_features=freq_feat.to_dict(),
            speaker_similarities={k: round(v, 4) for k, v in sims.items()},
        )

    # ── 批量处理 input/ 文件夹 ───────────────────────────────────────

    def process_input_folder(self, input_dir: Path, output_path: Optional[Path] = None) -> List[dict]:
        """处理 input/ 文件夹中的所有音频，返回 JSON 结果列表"""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        self.load_models()
        audio_files = sorted(
            [f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_AUDIO]
        )

        results = []
        for audio_file in audio_files:
            try:
                y, sr = self.load_audio(audio_file)
                y_16k, _ = librosa.load(str(audio_file), sr=16000, mono=True)
                result = self.recognize_and_detect(y_16k.astype(np.float32), 16000)

                time_feat = extract_time_features(y, sr)
                freq_feat = extract_freq_features(y, sr)

                entry = {
                    "file": str(audio_file.relative_to(input_dir)),
                    "file_absolute": str(audio_file.resolve()),
                    "original_sr": sr,
                    "duration_sec": round(len(y) / sr, 3),
                    "time_domain_features": time_feat.to_dict(),
                    "frequency_domain_features": freq_feat.to_dict(),
                    "speaker": result.speaker,
                    "similarity": round(result.similarity, 4),
                    "is_genuine": result.is_genuine,
                    "tts_score": round(result.tts_score, 4),
                    "asr_paraformer": result.paraformer_text or "",
                    "speaker_similarities": result.speaker_similarities,
                }
                results.append(entry)
            except Exception as exc:
                results.append({
                    "file": str(audio_file.relative_to(input_dir)),
                    "file_absolute": str(audio_file.resolve()),
                    "error": str(exc),
                })

        output_json = json.dumps(results, ensure_ascii=False, indent=2)
        if output_path:
            output_path.write_text(output_json, encoding="utf-8")

        return results

    # ── 评估 ─────────────────────────────────────────────────────────

    def evaluate_accuracy(
        self, test_dir: Path, ground_truth_file: Optional[Path] = None
    ) -> dict:
        """
        评估说话人识别准确率。
        目录结构: test_dir/speaker_name/*.wav
        可选: ground_truth_file (JSON) 每行 {file: speaker}
        """
        self.load_models()
        if not self.speaker_db:
            return {"error": "Speaker DB is empty. Please enroll first."}

        total = 0
        correct = 0
        details = []

        # 如果提供了 ground truth
        gt_map = {}
        if ground_truth_file and ground_truth_file.exists():
            gt_data = json.loads(ground_truth_file.read_text(encoding="utf-8"))
            if isinstance(gt_data, list):
                for item in gt_data:
                    gt_map[item["file"]] = item.get("speaker", "") or item.get("label", "")
            elif isinstance(gt_data, dict):
                gt_map = gt_data

        if gt_map:
            # 使用 ground truth 评估
            for audio_path_str, true_speaker in gt_map.items():
                audio_path = Path(audio_path_str)
                if not audio_path.exists():
                    continue
                try:
                    y, sr = self.load_audio(audio_path)
                    result = self.recognize_and_detect(y, sr)
                    total += 1
                    is_correct = result.speaker == true_speaker
                    if is_correct:
                        correct += 1
                    details.append({
                        "file": audio_path_str,
                        "true_speaker": true_speaker,
                        "predicted_speaker": result.speaker,
                        "correct": is_correct,
                        "similarity": round(result.similarity, 4),
                    })
                except Exception as exc:
                    details.append({
                        "file": audio_path_str,
                        "error": str(exc),
                    })
        else:
            # 按目录结构评估
            for speaker_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                true_speaker = speaker_dir.name
                files = [f for f in speaker_dir.rglob("*") if f.suffix.lower() in SUPPORTED_AUDIO]
                for audio_file in files:
                    try:
                        y, sr = self.load_audio(audio_file)
                        result = self.recognize_and_detect(y, sr)
                        total += 1
                        is_correct = result.speaker == true_speaker
                        if is_correct:
                            correct += 1
                        details.append({
                            "file": str(audio_file),
                            "true_speaker": true_speaker,
                            "predicted_speaker": result.speaker,
                            "correct": is_correct,
                            "similarity": round(result.similarity, 4),
                        })
                    except Exception as exc:
                        details.append({
                            "file": str(audio_file),
                            "error": str(exc),
                        })

        accuracy = round(correct / total, 4) if total > 0 else 0.0
        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "accuracy_pct": f"{accuracy * 100:.2f}%",
            "details": details,
        }
