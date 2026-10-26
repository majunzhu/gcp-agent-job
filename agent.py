"""
编排层：LangChain 1.x + LangGraph ReAct Agent。
给 Agent 一个目标，它会自己规划调用哪些工具（脚本→分镜→配音→画面→合成）。
新版 langchain(1.x) 已废弃 AgentExecutor，改用 langgraph.prebuilt.create_react_agent。
"""
from langgraph.prebuilt import create_react_agent

import config
from tools import (
    gen_script, gen_storyboard, tts, gen_image, gen_video, merge_video, get_llm,
)

TOOLS = [gen_script, gen_storyboard, tts, gen_image, gen_video, merge_video]

SYSTEM_PROMPT = """你是一个抖音短视频制作智能体。用户给你一个主题，你要按流程制作成片：

步骤：
1. gen_script 生成口播脚本
2. gen_storyboard 拆成分镜表（JSON）
3. 对每个分镜调 gen_image / gen_video 生成画面素材
4. tts 给旁白配音
5. merge_video 用 ffmpeg 合成最终成片

注意：文生图/文生视频当前可能返回 NotImplementedError，
此时如实告诉用户"画面环节待接入 API"，但脚本、分镜、配音、合成链路可以继续跑通。
最后给用户一段中文总结：成片路径、以及哪些环节待接入。"""


def build_agent():
    llm = get_llm()
    return create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def run_agent(agent, user_input: str) -> str:
    """LangGraph agent 通过 messages 调用，返回最终助手消息。"""
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    last = result["messages"][-1]
    content = getattr(last, "content", last)
    if isinstance(content, list):  # 新版可能返回多段内容
        content = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return content
