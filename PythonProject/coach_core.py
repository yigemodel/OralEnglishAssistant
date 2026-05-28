from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 优化原因：避免把敏感配置“写死”在代码里，便于部署与维护；
# 同时保留默认值不变，确保不改变现有业务行为/外部调用方式。
_DEFAULT_API_KEY = "sk-abb219229cb946c081c1e29ff1c59dce"

Mode = Literal["daily", "topic", "exam"]

_PROMPT_DAILY = """
你是英语口语日常生活陪练老师。
要求：
1) 先用简短英文自然回复并追问1个问题，鼓励用户多说。
2) 如果用户表达有明显问题，给出一条改正句（英文）。
3) 最后补一条中文学习建议（不超过30字）。
输出格式：
REPLY: ...
CORRECTION: ...
SUGGESTION: ...
"""

_PROMPT_TOPIC = """
你是英语口语话题讨论教练。
要求：
1) 围绕指定话题给出简短观点，并提出1-2个追问（英文）。
2) 给出1条更地道表达（英文）。
3) 用中文给1条改进建议。
输出格式：
REPLY: ...
CORRECTION: ...
SUGGESTION: ...
"""

_PROMPT_EXAM = """
你是IELTS Speaking考官（模拟考试模式）。
规则：
1) 严格按Part1->Part2->Part3推进。
2) 当前轮先提出一个问题（英文）。
3) 如已有用户回答，则给出简短反馈：1条改进表达（英文）+1条中文建议。
4) 输出格式固定：
REPLY: ...
CORRECTION: ...
SUGGESTION: ...
"""

_PROMPT_EXAM_REPORT = """
你是雅思口语评分助手。请根据对话记录给出估算评分。
严格按如下格式输出，每行一个字段：
FLUENCY: <0-9分，可小数>
LEXICAL: <0-9分，可小数>
GRAMMAR: <0-9分，可小数>
PRONUNCIATION: <0-9分，可小数>
OVERALL: <0-9分，可小数>
COMMENT_EN: <英文2-3句>
COMMENT_CN: <中文2-3句>
"""


class CoachState(TypedDict):
    user_input: str
    mode: Mode
    topic: str
    level: str
    exam_part: int
    exam_round: int
    history: List[str]
    response: str
    correction: str
    suggestion: str
    forced_mode: Literal["auto", "daily", "topic", "exam"]
    exam_finished: bool
    exam_score: Dict[str, float]
    exam_comment_cn: str
    exam_comment_en: str


@dataclass
class DeepSeekClient:
    model: str = "deepseek-chat"
    temperature: float = 0.6

    def __post_init__(self) -> None:
        # 优化原因：允许通过环境变量覆盖 key，便于本地/部署切换；
        # 仍保留默认值，避免改变现有脚本直接运行时的行为。
        api_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or _DEFAULT_API_KEY
        )
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=self.temperature,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        result = self.llm.invoke(messages)
        if isinstance(result, AIMessage):
            return str(result.content).strip()
        return str(result).strip()


def create_initial_state(level: str = "B1") -> CoachState:
    return {
        "user_input": "",
        "mode": "daily",
        "topic": "general topic",
        "level": level,
        "exam_part": 0,
        "exam_round": 0,
        "history": [],
        "response": "",
        "correction": "",
        "suggestion": "",
        "forced_mode": "auto",
        "exam_finished": False,
        "exam_score": {},
        "exam_comment_cn": "",
        "exam_comment_en": "",
    }


def parse_mode(text: str) -> Mode:
    lowered = text.lower()
    exam_words = [
        "考试",
        "模拟",
        "ielts",
        "test",
        "exam",
        "part 1",
        "part 2",
        "part 3",
        "cue card",
        "考官",
    ]
    topic_words = ["话题", "讨论", "topic", "discuss", "debate", "观点"]

    if any(word in lowered for word in exam_words):
        return "exam"
    if any(word in lowered for word in topic_words):
        return "topic"
    return "daily"


def extract_topic(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "daily life"
    markers = ["topic:", "话题：", "话题:", "主题：", "主题:"]
    for marker in markers:
        idx = stripped.lower().find(marker.lower())
        if idx != -1:
            return stripped[idx + len(marker):].strip() or "general topic"
    return "general topic"


def split_answer(text: str) -> tuple[str, str, str]:
    response = ""
    correction = ""
    suggestion = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("REPLY:"):
            response = line[len("REPLY:"):].strip()
        elif line.upper().startswith("CORRECTION:"):
            correction = line[len("CORRECTION:"):].strip()
        elif line.upper().startswith("SUGGESTION:"):
            suggestion = line[len("SUGGESTION:"):].strip()
    if not response:
        response = text.strip()
    return response, correction, suggestion


def _apply_agent_reply(state: CoachState, text: str) -> None:
    # 优化原因：集中解析与写回逻辑，减少重复代码，降低后续修改风险。
    state["response"], state["correction"], state["suggestion"] = split_answer(text)


def _run_agent_and_update(
    state: CoachState,
    client: DeepSeekClient,
    *,
    system_prompt: str,
    user_prompt: str,
    history_tag: str,
) -> CoachState:
    # 优化原因：三类 agent 节点共有相同模板（prompt -> chat -> split -> history），抽取为通用函数。
    text = client.chat(system_prompt, user_prompt)
    _apply_agent_reply(state, text)
    state["history"].append(f"{history_tag} {state['user_input']}")
    return state


def router_node(state: CoachState) -> CoachState:
    if state["forced_mode"] != "auto":
        mode = state["forced_mode"]
    else:
        mode = parse_mode(state["user_input"])
    state["mode"] = mode
    if mode == "topic":
        state["topic"] = extract_topic(state["user_input"])
    return state


def daily_agent_node(state: CoachState, client: DeepSeekClient) -> CoachState:
    user_prompt = f"用户输入：{state['user_input']}\n用户水平：{state['level']}"
    return _run_agent_and_update(
        state,
        client,
        system_prompt=_PROMPT_DAILY,
        user_prompt=user_prompt,
        history_tag="[daily]",
    )


def topic_agent_node(state: CoachState, client: DeepSeekClient) -> CoachState:
    user_prompt = (
        f"当前话题：{state['topic']}\n"
        f"用户输入：{state['user_input']}\n"
        f"用户水平：{state['level']}"
    )
    return _run_agent_and_update(
        state,
        client,
        system_prompt=_PROMPT_TOPIC,
        user_prompt=user_prompt,
        history_tag=f"[topic:{state['topic']}]",
    )


def _ensure_exam_started(state: CoachState) -> tuple[int, int]:
    # 优化原因：将“首次进入 exam 时初始化 part/round”的逻辑从主函数中抽离。
    if state["exam_part"] == 0:
        state["exam_part"] = 1
        state["exam_round"] = 1
    return state["exam_part"], state["exam_round"]


def _advance_exam_state(state: CoachState) -> None:
    # 优化原因：集中管理考试状态推进规则，便于阅读与单点修改，避免散落条件分支。
    state["exam_round"] += 1
    if state["exam_part"] == 1 and state["exam_round"] > 4:
        state["exam_part"] = 2
        state["exam_round"] = 1
    elif state["exam_part"] == 2 and state["exam_round"] > 2:
        state["exam_part"] = 3
        state["exam_round"] = 1
    elif state["exam_part"] == 3 and state["exam_round"] > 4:
        state["exam_finished"] = True


def exam_agent_node(state: CoachState, client: DeepSeekClient) -> CoachState:
    state["exam_finished"] = False
    part, round_idx = _ensure_exam_started(state)
    user_prompt = (
        f"当前Part：{part}\n"
        f"当前轮次：{round_idx}\n"
        f"用户输入：{state['user_input']}\n"
        f"用户水平：{state['level']}"
    )
    text = client.chat(_PROMPT_EXAM, user_prompt)
    _apply_agent_reply(state, text)

    _advance_exam_state(state)
    if state["exam_finished"]:
        generate_exam_report(state, client)
        state["response"] += "\n\nMock exam finished. Please click restart to start over."

    state["history"].append(
        f"[exam-part{part}-round{round_idx}] {state['user_input']}"
    )
    return state


def generate_exam_report(state: CoachState, client: DeepSeekClient) -> None:
    history_text = "\n".join(state["history"][-20:])
    user_prompt = f"用户水平：{state['level']}\n最近考试对话：\n{history_text}"
    text = client.chat(_PROMPT_EXAM_REPORT, user_prompt)
    score, en_comment, cn_comment = parse_exam_report(text)
    state["exam_score"] = score
    state["exam_comment_en"] = en_comment
    state["exam_comment_cn"] = cn_comment


def parse_exam_report(text: str) -> tuple[Dict[str, float], str, str]:
    fields = {
        "FLUENCY": 0.0,
        "LEXICAL": 0.0,
        "GRAMMAR": 0.0,
        "PRONUNCIATION": 0.0,
        "OVERALL": 0.0,
    }
    en_comment = ""
    cn_comment = ""
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in fields:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = 0.0
        elif key == "COMMENT_EN":
            en_comment = value
        elif key == "COMMENT_CN":
            cn_comment = value
    return fields, en_comment, cn_comment


def run_multi_agent_once(state: CoachState, client: DeepSeekClient) -> CoachState:
    state = router_node(state)
    if state["mode"] == "daily":
        return daily_agent_node(state, client)
    if state["mode"] == "topic":
        return topic_agent_node(state, client)
    return exam_agent_node(state, client)
