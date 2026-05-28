"""Streamlit Web 界面：支持文本 / 语音两种交流方式，可随时切换。"""

import base64
import sys
from pathlib import Path


def _ensure_streamlit_runtime() -> None:
    if __name__ != "__main__":
        return
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            return
    except Exception:
        # 优化原因：保持原有“兜底启动”行为不变；这里不抛错是为了兼容不同 Streamlit 版本/运行环境。
        pass

    import subprocess

    app = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    print("\n检测到未使用 streamlit run，正在自动启动 Web 服务…")
    print("浏览器地址一般为: http://localhost:8501\n")
    raise SystemExit(subprocess.call(cmd))


_ensure_streamlit_runtime()

import streamlit as st
import streamlit.components.v1 as components

from coach_core import DeepSeekClient, create_initial_state, run_multi_agent_once
from voice_utils import (
    extract_english_for_tts,
    text_to_speech_bytes,
    transcribe_audio,
)

st.set_page_config(
    page_title="英语口语陪练",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

VOICE_OPTIONS = {
    "英语（美）Jenny": "en-US-JennyNeural",
    "英语（英）Ryan": "en-GB-RyanNeural",
    "英语（澳）Natasha": "en-AU-NatashaNeural",
}


def format_bot_reply(new_state: dict) -> str:
    lines = [new_state["response"]]
    if new_state["correction"]:
        lines.append(f"**Correction:** {new_state['correction']}")
    if new_state["suggestion"]:
        lines.append(f"**Suggestion:** {new_state['suggestion']}")
    lines.append(
        f"`模式: {new_state['mode']} | 强制: {new_state['forced_mode']} | "
        f"Exam P{new_state['exam_part']} R{new_state['exam_round']}`"
    )
    return "\n\n".join(lines)


def _init_session_state() -> None:
    # 优化原因：集中 session_state 初始化，避免散落多处导致 key 漏配或行为漂移。
    defaults = {
        "coach_state": create_initial_state(level="B1"),
        "messages": [],
        "client": DeepSeekClient(),
        "chat_mode": "文本输入",
        "auto_play_voice": True,
        "tts_voice": VOICE_OPTIONS["英语（美）Jenny"],
        "last_reply_audio": None,
        "last_tts_error": None,
        "stt_engine": "sphinx",
        "voice_transcript_box": "",
        "clear_voice_box_next_run": False,
        "pending_transcript": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_conversation(*, level: str, forced_mode: str) -> None:
    # 优化原因：集中“清空会话”逻辑，保持与 UI 选择同步，减少重复代码。
    st.session_state.coach_state = create_initial_state(level=level)
    st.session_state.coach_state["forced_mode"] = forced_mode
    st.session_state.messages = []
    st.session_state.last_reply_audio = None


def _render_autoplay_audio_mp3(audio_bytes: bytes, *, key: str) -> None:
    # 优化原因：Streamlit 的 st.audio 通常需要用户再点一次“播放”；
    # 这里用浏览器原生 audio autoplay，让“点击播放按钮”后立即播放，体验更像微信语音。
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    html = f"""
    <audio id="{key}" autoplay="autoplay">
      <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>
      (function() {{
        const el = document.getElementById("{key}");
        if (!el) return;
        const p = el.play();
        if (p && p.catch) p.catch(() => {{}});
      }})();
    </script>
    """
    components.html(html, height=0)


def _ensure_assistant_msg_audio(idx: int, content: str) -> None:
    # 优化原因：为了实现“微信气泡：一次点击即播放”，需要提前准备音频数据，
    # 避免依赖 st.audio 控件（它常要求用户二次点击播放器）。
    audio_key = f"msg_audio_{idx}"
    err_key = f"msg_audio_err_{idx}"
    if audio_key in st.session_state or err_key in st.session_state:
        return

    english = extract_english_for_tts(content)
    audio_bytes, err = text_to_speech_bytes(english, voice=st.session_state.tts_voice)
    if audio_bytes:
        st.session_state[audio_key] = audio_bytes
        st.session_state[err_key] = ""
    else:
        st.session_state[err_key] = err or "语音合成失败"


def _render_wechat_voice_bubble(*, audio_bytes: bytes, bubble_id: str) -> None:
    # 优化原因：模仿微信语音气泡（单按钮切换播放/暂停 + 播放中动画），并在按钮点击事件中直接触发播放。
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    html = f"""
    <style>
      .wx-bubble {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,.08);
        background: #95EC69; /* 微信绿 */
        color: #111;
        cursor: pointer;
        user-select: none;
        font-size: 14px;
        line-height: 1;
        box-shadow: 0 1px 0 rgba(0,0,0,.05);
      }}
      .wx-bubble:active {{
        transform: translateY(1px);
      }}
      .wx-icon {{
        width: 14px;
        height: 14px;
        display: inline-block;
      }}
      .wx-bars {{
        display: inline-flex;
        gap: 2px;
        align-items: flex-end;
        height: 12px;
        width: 18px;
      }}
      .wx-bars span {{
        display: inline-block;
        width: 3px;
        height: 4px;
        background: rgba(0,0,0,.55);
        border-radius: 2px;
        opacity: 1;
      }}
      .wx-playing .wx-bars span {{
        animation: wxbar 0.9s ease-in-out infinite;
      }}
      .wx-playing .wx-bars span:nth-child(2) {{ animation-delay: .12s; }}
      .wx-playing .wx-bars span:nth-child(3) {{ animation-delay: .24s; }}
      @keyframes wxbar {{
        0% {{ height: 3px; }}
        50% {{ height: 12px; }}
        100% {{ height: 3px; }}
      }}
    </style>

    <div>
      <button id="btn-{bubble_id}" class="wx-bubble" type="button" aria-label="播放/暂停">
        <span class="wx-icon">🔊</span>
        <span class="wx-bars" aria-hidden="true">
          <span></span><span></span><span></span>
        </span>
      </button>
      <audio id="aud-{bubble_id}">
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
      </audio>
    </div>

    <script>
      (function() {{
        const btn = document.getElementById("btn-{bubble_id}");
        const aud = document.getElementById("aud-{bubble_id}");
        if (!btn || !aud) return;

        function setPlaying(isPlaying) {{
          if (isPlaying) {{
            btn.classList.add("wx-playing");
          }} else {{
            btn.classList.remove("wx-playing");
          }}
        }}

        aud.addEventListener("ended", () => setPlaying(false));
        aud.addEventListener("pause", () => setPlaying(false));
        aud.addEventListener("play", () => setPlaying(true));

        btn.addEventListener("click", async () => {{
          try {{
            if (aud.paused) {{
              await aud.play(); // 关键：在按钮点击手势内直接播放，避免二次点击
            }} else {{
              aud.pause();
            }}
          }} catch (e) {{
            // 某些浏览器策略下可能阻止播放；保持按钮可点击，不弹出系统播放器菜单
            setPlaying(false);
          }}
        }});
      }})();
    </script>
    """
    components.html(html, height=54)


def render_chat_message(msg: dict, idx: int) -> None:
    """渲染单条消息；助手消息支持就地语音播放按钮。"""
    with st.chat_message(msg["role"]):
        if msg["role"] != "assistant":
            st.markdown(msg["content"])
            return

        text_col, bubble_col = st.columns([0.78, 0.22], gap="small")
        with text_col:
            st.markdown(msg["content"])
        with bubble_col:
            _ensure_assistant_msg_audio(idx, msg["content"])
            audio_data = st.session_state.get(f"msg_audio_{idx}")
            if audio_data:
                _render_wechat_voice_bubble(
                    audio_bytes=audio_data,
                    bubble_id=f"m{idx}",
                )
        audio_err = st.session_state.get(f"msg_audio_err_{idx}")
        if audio_err:
            st.warning(audio_err)


def handle_user_message(user_text: str) -> None:
    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    state = st.session_state.coach_state
    state["user_input"] = user_text

    with st.spinner("陪练老师思考中…"):
        new_state = run_multi_agent_once(state, st.session_state.client)
    st.session_state.coach_state = new_state

    bot_text = format_bot_reply(new_state)
    st.session_state.messages.append({"role": "assistant", "content": bot_text})
    # 优化原因：提前为该条助手消息生成语音，确保对话记录里的“微信气泡”可以一次点击直接播放。
    assistant_idx = len(st.session_state.messages) - 1
    _ensure_assistant_msg_audio(assistant_idx, bot_text)

    if st.session_state.chat_mode == "语音对话" and st.session_state.auto_play_voice:
        english = extract_english_for_tts(new_state["response"])
        audio_bytes, err = text_to_speech_bytes(
            english, voice=st.session_state.tts_voice
        )
        if audio_bytes:
            st.session_state.last_reply_audio = audio_bytes
        else:
            st.session_state.last_reply_audio = None
            st.session_state.last_tts_error = err
    else:
        st.session_state.last_reply_audio = None


_init_session_state()

# ---------- 顶栏 ----------
st.title("🎙️ 英语口语陪练助手")
st.caption("练习对话使用英语 · 纠错与讲解使用中文 ·     文本 / 语音可随时切换")

mode_cols = st.columns([1, 1, 2])
with mode_cols[0]:
    if st.button(
        "📝 文本输入",
        use_container_width=True,
        type="primary" if st.session_state.chat_mode == "文本输入" else "secondary",
    ):
        st.session_state.chat_mode = "文本输入"
        st.rerun()
with mode_cols[1]:
    if st.button(
        "💬 语音对话",
        use_container_width=True,
        type="primary" if st.session_state.chat_mode == "语音对话" else "secondary",
    ):
        st.session_state.chat_mode = "语音对话"
        st.rerun()
with mode_cols[2]:
    st.info(f"当前交流方式：**{st.session_state.chat_mode}**")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("练习设置")
    level = st.selectbox("英语水平", ["A2", "B1", "B2", "C1"], index=1)
    st.session_state.coach_state["level"] = level

    force_mode = st.selectbox(
        "练习模块",
        ["auto", "daily", "topic", "exam"],
        index=["auto", "daily", "topic", "exam"].index(
            st.session_state.coach_state.get("forced_mode", "auto")
        ),
        format_func=lambda x: {
            "auto": "自动识别",
            "daily": "日常生活",
            "topic": "话题讨论",
            "exam": "口语考试",
        }[x],
    )
    st.session_state.coach_state["forced_mode"] = force_mode
    if force_mode == "topic":
        topic = st.text_input("话题", value=st.session_state.coach_state["topic"])
        st.session_state.coach_state["topic"] = topic or "general topic"

    st.divider()
    st.subheader("语音设置")
    st.session_state.auto_play_voice = st.toggle(
        "自动朗读英文回复", value=st.session_state.auto_play_voice
    )
    voice_label = st.selectbox("朗读音色", list(VOICE_OPTIONS.keys()))
    st.session_state.tts_voice = VOICE_OPTIONS[voice_label]

    stt_label = st.selectbox(
        "识别引擎",
        ["sphinx", "auto", "google"],
        index=["sphinx", "auto", "google"].index(st.session_state.stt_engine),
        format_func=lambda x: {
            "sphinx": "本地离线识别（PocketSphinx，推荐）",
            "auto": "自动（先离线，再 Google）",
            "google": "Google 在线（易超时）",
        }[x],
    )
    st.session_state.stt_engine = stt_label
    st.caption("识别：PocketSphinx(离线) / Google(在线) · 朗读：edge-tts")

    st.divider()
    if st.button("清空会话", use_container_width=True):
        _clear_conversation(level=level, forced_mode=force_mode)
        st.success("已清空")

# ---------- 主区：对话历史 ----------
chat_col, input_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader("对话记录")
    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                "_暂无消息。请在右侧用 **文本** 或 **语音** 开始练习。_"
            )
        for i, msg in enumerate(st.session_state.messages):
            render_chat_message(msg, i)

    if st.session_state.coach_state.get("exam_finished"):
        st.subheader("📊 考试评分卡")
        score = st.session_state.coach_state.get("exam_score", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Fluency", f"{score.get('FLUENCY', 0):.1f}")
        m2.metric("Lexical", f"{score.get('LEXICAL', 0):.1f}")
        m3.metric("Grammar", f"{score.get('GRAMMAR', 0):.1f}")
        m4.metric("Pronunciation", f"{score.get('PRONUNCIATION', 0):.1f}")
        m5.metric("Overall", f"{score.get('OVERALL', 0):.1f}")
        if st.session_state.coach_state.get("exam_comment_cn"):
            st.markdown(st.session_state.coach_state["exam_comment_cn"])

with input_col:
    st.subheader("输入区")

    if st.session_state.chat_mode == "文本输入":
        st.markdown("##### 📝 文本交流")
        text_prompt = st.text_area(
            "输入英语或中文指令",
            height=120,
            placeholder="例如：Let's practice daily conversation about shopping.",
            key="text_area_input",
        )
        if st.button("发送", type="primary", use_container_width=True):
            handle_user_message(text_prompt)
            st.rerun()

    else:
        st.markdown("##### 💬 语音交流")
        st.caption("点击麦克风录音 → 识别为文字 → 发送给陪练 → 可自动朗读回复")

        if st.session_state.clear_voice_box_next_run:
            st.session_state.voice_transcript_box = ""
            st.session_state.clear_voice_box_next_run = False

        audio = st.audio_input("录制你的回答（英语）", key="voice_recorder")

        if audio is not None and st.button("🔊 识别语音", use_container_width=True):
            hint = (
                "正在进行本地离线识别…"
                if st.session_state.stt_engine in ("sphinx", "auto")
                else "正在连接 Google 识别…"
            )
            with st.spinner(hint):
                text, err = transcribe_audio(
                    audio.getvalue(),
                    engine=st.session_state.stt_engine,
                )
            if err:
                st.error(err)
                st.info(
                    "排查建议：1) 先点“语音链路自检”；2) 录制 2-6 秒纯英文；"
                    "3) 引擎改为 local/auto；4) 识别失败可先手动编辑文本再发送。"
                )
            else:
                st.session_state.pending_transcript = text
                st.session_state.voice_transcript_box = text
                st.success(f"识别成功：{text}")
                st.rerun()

        if (
            st.session_state.pending_transcript
            and not st.session_state.voice_transcript_box
        ):
            st.session_state.voice_transcript_box = st.session_state.pending_transcript

        transcript = st.text_area(
            "识别结果（可编辑后发送）",
            height=100,
            key="voice_transcript_box",
        )

        col_send = st.columns(1)[0]
        with col_send:
            if st.button("发送识别文本", type="primary", use_container_width=True):
                handle_user_message(transcript)
                st.session_state.pending_transcript = ""
                st.session_state.clear_voice_box_next_run = True
                st.rerun()

        if st.session_state.last_reply_audio:
            # 优化原因：避免渲染原生播放器控件导致“发送后唤醒播放器/需二次点击”的体验；
            # 这里改为隐藏自动播放，与对话记录气泡一致做到“一次操作即播放”。
            _render_autoplay_audio_mp3(
                st.session_state.last_reply_audio, key="last_reply_autoplay"
            )
        if st.session_state.get("last_tts_error"):
            st.warning(st.session_state.last_tts_error)

    st.divider()
