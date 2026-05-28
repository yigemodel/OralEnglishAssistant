"""语音工具：录音转文字（STT）与回复朗读（TTS）。

稳定策略：
1) 始终先把浏览器录音转成 16k/mono wav
2) 优先用 PocketSphinx 本地离线识别（无需外网）
3) 本地失败再尝试 Google 在线识别（可选）
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Literal

SttEngine = Literal["auto", "sphinx", "google"]

_GOOGLE_TIMEOUT_SEC = 8
_FFMPEG_CONVERT_TIMEOUT_SEC = 90
_MIN_WAV_SIZE_BYTES = 16000

def extract_english_for_tts(markdown_text: str) -> str:
    """从助手回复中提取适合朗读的英文（优先 REPLY 段或首段英文）。"""
    for line in markdown_text.splitlines():
        line = line.strip()
        if line.upper().startswith("REPLY:"):
            return line[6:].strip()
    plain = re.sub(r"\*\*[^*]+\*\*", "", markdown_text)
    for line in plain.splitlines():
        line = line.strip()
        if line and not line.startswith("[") and not line.startswith("`"):
            if re.search(r"[A-Za-z]", line):
                return line[:500]
    return "Please continue your practice."


def _get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def _is_wav(audio_bytes: bytes) -> bool:
    return len(audio_bytes) > 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _convert_to_wav(audio_bytes: bytes) -> tuple[Path | None, list[Path], str | None]:
    temps: list[Path] = []

    if _is_wav(audio_bytes):
        wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        wav_path.write_bytes(audio_bytes)
        temps.append(wav_path)
        return wav_path, temps, None

    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        return (
            None,
            temps,
            "未找到 ffmpeg。请执行: pip install imageio-ffmpeg",
        )

    inp_path = Path(tempfile.mkstemp(suffix=".webm")[1])
    inp_path.write_bytes(audio_bytes)
    temps.append(inp_path)

    wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    temps.append(wav_path)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(inp_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        str(wav_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_FFMPEG_CONVERT_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, temps, "音频转换超时，请缩短录音后重试。"
    except OSError as exc:
        return None, temps, f"无法调用 ffmpeg: {exc}"

    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 44:
        err = (result.stderr or result.stdout or "未知错误").strip()
        return None, temps, f"音频格式转换失败: {err[:300]}"

    return wav_path, temps, None


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _transcribe_sphinx_local(wav_path: Path, language: str = "en-US") -> tuple[str, str | None]:
    """离线识别：PocketSphinx。"""
    try:
        import speech_recognition as sr
    except ImportError:
        return "", "缺少 SpeechRecognition，请执行: pip install SpeechRecognition"

    try:
        import pocketsphinx  # noqa: F401
    except ImportError:
        return "", "缺少 pocketsphinx，请执行: pip install pocketsphinx"

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            recorded = recognizer.record(source)
        # language 参数可忽略，默认英文模型
        text = recognizer.recognize_sphinx(recorded, language="en-US")
        text = text.strip()
        if not text:
            return "", "本地识别未识别到有效英文语音。"
        return text, None
    except sr.UnknownValueError:
        return "", "本地识别未听清，请录制更清晰的英文语音。"
    except sr.RequestError as exc:
        return "", f"PocketSphinx 配置异常: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "", f"本地识别失败: {exc}"


def _transcribe_google(wav_path: Path, language: str = "en-US") -> tuple[str, str | None]:
    try:
        import speech_recognition as sr
    except ImportError:
        return "", "缺少 SpeechRecognition，请执行: pip install SpeechRecognition"

    def _do_recognize() -> str:
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            recorded = recognizer.record(source)
        return recognizer.recognize_google(recorded, language=language)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_recognize)
            text = future.result(timeout=_GOOGLE_TIMEOUT_SEC)
        return text.strip(), None
    except FuturesTimeoutError:
        return (
            "",
            "Google 语音识别连接超时（国内网络常见）。"
            # 优化原因：原文案提到 Whisper，但本项目当前并未实现该引擎，避免误导用户。
            "请在侧边栏将「识别引擎」改为「本地离线识别（PocketSphinx）」或「auto」。",
        )
    except sr.UnknownValueError:
        return "", "未能识别语音，请靠近麦克风、用英语清晰复述。"
    except sr.RequestError as exc:
        return (
            "",
            f"Google 语音识别不可用: {exc}。"
            # 优化原因：同上，纠正文案以匹配实际支持的引擎选项。
            "建议改用侧边栏「本地离线识别（PocketSphinx）」或「auto」。",
        )
    except Exception as exc:  # noqa: BLE001
        return "", f"Google 识别失败: {exc}"


def transcribe_audio(
    audio_bytes: bytes,
    language: str = "en-US",
    engine: SttEngine = "auto",
) -> tuple[str, str | None]:
    """
    将浏览器录音转为文本。
    engine:
      - sphinx: 本地 PocketSphinx（推荐，不依赖外网）
      - google: Google 在线（可能超时）
      - auto: 先本地，再尝试 Google
    """
    if not audio_bytes:
        return "", "未收到音频，请先录音。"

    wav_path, temps, conv_err = _convert_to_wav(audio_bytes)
    if conv_err or wav_path is None:
        _cleanup(temps)
        return "", conv_err or "音频转换失败。"

    try:
        # 录音过短时识别稳定性很差，提前提示
        if wav_path.exists() and wav_path.stat().st_size < _MIN_WAV_SIZE_BYTES:
            return "", "录音过短或无有效声音，请至少录制 1-2 秒再试。"

        # 优化原因：合并重复的“先本地后在线”兜底分支，避免逻辑漂移与重复代码。
        if engine == "google":
            return _transcribe_google(wav_path, language=language)

        text, local_err = _transcribe_sphinx_local(wav_path, language=language)
        if text:
            return text, None

        # engine 为 sphinx 或 auto 时：本地失败后尝试 Google 兜底（与原行为一致）。
        text2, google_err = _transcribe_google(wav_path, language=language)
        if text2:
            return text2, None
        return "", local_err or google_err or "语音识别失败，请改用手动输入文本。"
    finally:
        _cleanup(temps)


async def _edge_tts_save(text: str, voice: str, out_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def text_to_speech_bytes(
    text: str, voice: str = "en-US-JennyNeural"
) -> tuple[bytes | None, str | None]:
    if not text.strip():
        return None, "没有可朗读的英文内容。"

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        return None, "缺少 edge-tts，请执行: pip install edge-tts"

    out_path = Path(tempfile.mkstemp(suffix=".mp3")[1])
    try:
        asyncio.run(_edge_tts_save(text, voice, out_path))
        return out_path.read_bytes(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"语音合成失败: {exc}"
    finally:
        out_path.unlink(missing_ok=True)
