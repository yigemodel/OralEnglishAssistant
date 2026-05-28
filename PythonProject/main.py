"""命令行入口：英语口语陪练多 Agent Demo（LangChain + DeepSeek）"""

from coach_core import DeepSeekClient, create_initial_state, run_multi_agent_once, CoachState


def pretty_print(state: CoachState) -> None:
    print("\n=== BOT ===")
    print(state["response"])
    if state["correction"]:
        print(f"\n[Correction] {state['correction']}")
    if state["suggestion"]:
        print(f"[Suggestion] {state['suggestion']}")
    if state["mode"] == "exam":
        print(f"[Exam State] part={state['exam_part']}, round={state['exam_round']}")
    print("===========\n")


def main() -> None:
    print("English Speaking Coach (LangChain + DeepSeek)")
    print("输入 'exit' 退出，输入 'restart exam' 可重置考试状态。")
    state = create_initial_state(level="B1")
    client = DeepSeekClient()
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() == "exit":
            break
        if user_text.lower() == "restart exam":
            state["exam_part"] = 0
            state["exam_round"] = 0
            print("考试状态已重置。")
            continue

        state["user_input"] = user_text
        state = run_multi_agent_once(state, client)  # 保留历史状态，形成多轮会话
        pretty_print(state)


if __name__ == "__main__":
    main()
