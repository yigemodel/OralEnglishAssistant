## English Speaking Coach（LangChain + DeepSeek）

一个英语口语陪练 Demo：支持 **CLI 交互** 与 **Streamlit Web 页面** 两种入口；Web 端支持 **文本/语音输入**，并可对助手回复进行 **一键语音播放（微信气泡样式）**。

---

## 运行环境

- **Python**：建议 \(>= 3.10\)（项目使用了 `X | Y` 类型标注）
- **依赖安装**：

```bash
pip install -r requirements.txt
```

---

## 启动方式

### CLI（命令行）

```bash
python main.py
```

- 输入 `exit` 退出
- 输入 `restart exam` 重置考试状态（CLI 入口保留原行为）

### Web（Streamlit）

推荐使用一键启动脚本：

```bash
python run_web.py
```

或直接：

```bash
streamlit run web_app.py
```

---

## 环境变量（可选）

`coach_core.py` 默认仍保留内置 key（为了不改变原行为），但支持使用环境变量覆盖，便于部署维护：

- `DEEPSEEK_API_KEY`：DeepSeek API Key（优先）
- `OPENAI_API_KEY`：兼容 `ChatOpenAI` 习惯变量（次优先）

示例：

```bash
export DEEPSEEK_API_KEY="your_key_here"
python run_web.py
```

---

## 文件说明（每个文件做什么）

- **`main.py`**
  - CLI 入口：`input()` 循环读取用户输入，调用核心逻辑并打印结果。

- **`run_web.py`**
  - Web 启动脚本：内部执行 `python -m streamlit run web_app.py`，方便一键启动。

- **`web_app.py`**
  - Streamlit Web UI：文本/语音输入、对话记录展示、语音播放。
  - 对话记录的语音播放采用 **微信气泡单按钮**：一次点击直接播放/暂停，播放中有动画。
  - 语音对话模式下自动朗读使用隐藏 `<audio autoplay>`，避免唤醒原生播放器控件导致二次点击。

- **`coach_core.py`**
  - 核心对话/模式路由：
    - `create_initial_state()`：初始化会话状态
    - `run_multi_agent_once()`：按模式路由到 daily/topic/exam
    - `DeepSeekClient`：通过 `ChatOpenAI(base_url="https://api.deepseek.com")` 调用 DeepSeek
  - exam 模式支持推进 Part1/2/3，并在结束后生成评分报告。

- **`voice_utils.py`**
  - 语音工具：
    - `transcribe_audio()`：浏览器录音 -> wav（ffmpeg 转码）-> STT（PocketSphinx 优先，Google 兜底）
    - `text_to_speech_bytes()`：edge-tts 合成 mp3 字节
    - `extract_english_for_tts()`：从助手回复中提取更适合朗读的英文片段

---

## 常见问题

- **点击播放为什么有时仍可能失败？**
  - 部分浏览器会对 autoplay 做限制；目前播放触发发生在按钮点击这一“用户手势”内，已尽量规避二次点击体验。

