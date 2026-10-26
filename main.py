"""入口：python main.py "你的主题" """
import sys

from agent import build_agent, run_agent


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI 如何帮我写周报"
    agent = build_agent()
    result = run_agent(agent, f"为主题「{topic}」制作一条抖音竖屏视频")
    print("\n" + "=" * 50)
    print(result)


if __name__ == "__main__":
    main()
