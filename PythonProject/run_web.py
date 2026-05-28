"""一键启动 Web 界面（内部调用 streamlit run）。"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).resolve().parent / "web_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    print("正在启动 Streamlit，浏览器将打开 http://localhost:8501")
    print("命令:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
