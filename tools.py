"""
工具层：LangChain 工具集合。
每个环节封装成一个 @tool，Agent 会按需求自动调用。
"""
import json
import os
import re
import subprocess
from pathlib import Path

from langchain.tools import tool
from langchain_openai import ChatOpenAI

import config


def get_ffmpeg():
    """优先用项目内置 bin/ffmpeg，否则回退系统 PATH 中的 ffmpeg。"""
    local = Path(__file__).parent / "bin" / "ffmpeg"
    if local.exists():
        return str(local)
    return "ffmpeg"

# 供所有工具复用的 LLM（脚本、分镜都用它）
def get_llm():
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.7,
    )


@tool
def gen_script(topic: str, content: str = "") -> str:
    """根据用户指定的主题和内容，生成一段 60~90 秒的竖屏口播脚本（含开头钩子、正文、结尾引导）。

    严格遵守以下约束：
    1. 主题（topic）是唯一允许的视频方向，严禁自行更换或新增主题。
    2. 若提供了内容（content），脚本必须严格围绕其中的信息创作：
       - 用户的每一条要点都必须被涵盖，不能遗漏；
       - 不得编造用户没有提到的核心事实、数据或结论；
       - 可做的仅是润色表达、组织顺序、口语化，不能改变原意；
       - 若用户内容本身就是完整素材，直接以它为脚本主体。
    3. 开头 3 秒抓眼球，中间信息密度高，结尾引导点赞关注；每句一行，口语化，含【画面:xxx】。
    """
    llm = get_llm()
    if content:
        prompt = (
            f"你是抖音短视频编导。\n"
            f"【视频标题/主题，必须严格遵守，不可偏离】\n{topic}\n\n"
            f"【用户提供的创作内容，必须严格以此为准进行创作，逐条覆盖，不得遗漏或编造】\n{content}\n\n"
            f"现在基于以上主题和内容，写一段 60~90 秒竖屏口播脚本。\n"
            "硬性要求：\n"
            "- 内容 100% 来源于上面的用户素材，只做口语化润色和排序，绝不添加用户没写的事实/数据/结论；\n"
            "- 用户内容的每一条都要出现在脚本里，可用开头钩子做引入；\n"
            "- 开头 3 秒抓眼球，结尾引导点赞关注；\n"
            "- 每句一行，口语化，含【画面:xxx】画面提示。\n"
            "直接输出脚本正文，不要解释。"
        )
    else:
        prompt = (
            f"你是抖音短视频编导。为主题「{topic}」写一段 60~90 秒竖屏口播脚本。\n"
            "要求：严格围绕该主题，不跑题、不编造与主题无关的内容；"
            "开头 3 秒抓眼球，中间信息密度高，结尾引导点赞关注；"
            "每句一行，口语化，含画面提示【画面:xxx】。直接输出脚本正文，不要解释。"
        )
    return llm.invoke(prompt).content


@tool
def gen_storyboard(script: str) -> str:
    """把口播脚本拆成分镜表，返回 JSON：每条 = 画面描述 + 旁白 + 时长。

    严格遵守：仅基于给定脚本拆解，旁白必须逐句取自脚本原文，不得改写、增删或补内容。"""
    llm = get_llm()
    prompt = (
        "把下面口播脚本严格拆成 4~8 个分镜，输出严格 JSON 数组，字段："
        '{"scene": 画面描述, "narration": 该镜旁白, "seconds": 时长}。\n'
        "硬性要求：\n"
        "- narration 必须逐句使用脚本原文，一字不改、不增删；\n"
        "- scene 只做画面提示的描述，不添加脚本没有的内容；\n"
        "- 所有分镜的 narration 拼接起来必须等于原脚本全部内容。\n"
        f"脚本：\n{script}\n只输出 JSON。"
    )
    raw = llm.invoke(prompt).content
    # 容错：去掉可能包裹的 ```json ... ```
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


@tool
def gen_review(topic: str, script: str, shots_json: str) -> str:
    """爆款级多角色质量把控：让 LLM 同时扮演抖音爆款视频专家、心理专家、
    视频导演、分镜设计专家、视频字幕专家、文案编辑六个角色，从停顿节奏、
    重点凸显、节奏流畅、视频美感、冲击力、情绪价值、吸引力等专业维度，
    审查现有脚本+分镜，输出优化后的分镜 JSON。

    硬约束：narration 必须逐句忠于原脚本核心内容（可微调口语化+加停顿标点，
    但不偏离原意、不增删核心事实）；返回的 JSON 含 pause/emphasis/tone 增强字段。"""
    llm = get_llm()
    prompt = (
        "你是由七名顶级专家组成的抖音爆款视频创作委员会，以极严苛、专业、一针见血的标准"
        "审查并优化下面的口播脚本和分镜。目标：必须达到抖音爆款头部水准，绝不允许平庸。\n"
        "【对标爆款标杆：抖音知识科普头部账号@鲁大猿 的风格基准，请学习并应用到分镜】\n"
        "- 内容定位：硬核知识科普(物理/AI/原理/方法论)，专业深度强、观点独到、有稀缺价值；\n"
        "- 开头钩子：3秒内用强烈惊叹或悬念切入(如'四力统一实现了!!!')，制造好奇缺口，绝不温吞开场；\n"
        "- 演进逻辑：用'层层递进+反转+对比'推进(先抛结论，再拆为什么，再给数据/依据，最后抛重磅钩子引导下期)；\n"
        "- 结构节奏：必须多用分条(1、2、3...)层层递进、逻辑清晰，每3~5秒一个节奏变化，避免平铺；\n"
        "- 标题要求：更硬核、更有冲击力、用惊叹/悬念/数字/强观点(如'别再瞎忙了!!!' '一条铁律搞定xx')，"
        "绝不温和平淡；\n"
        "- 特定配色：整体统一协调的高级配色——背景明亮多彩、标题用醒目金色/亮黄、内容字幕纯白"
        "配深描边保证高对比可读，视觉有设计感不杂乱；\n"
        "- 内容密度：信息密度高、说人话、短句有力、用具体数字/场景佐证，拒绝空话套话；\n"
        "- 情绪价值：把枯燥知识讲出'恍然大悟'的爽感，结尾制造悬念或价值感引导关注；\n"
        "- 视觉呈现：画面辅助讲解、重点突出、冲击力强、图文并茂，符合竖屏爆款审美。\n\n"
        "【视频标题/主题】\n" + topic + "\n\n"
        "【现有分镜 JSON】\n" + shots_json + "\n\n"
        "七位专家职责与审查维度（每位必须从专业角度找出问题并优化，禁止空泛好评）：\n"
        "- 抖音爆款视频专家：审查开头 3 秒钩子是否足够抓人、是否有反转/悬念/金句、"
        "每 3~5 秒一个节奏变化、结尾引导转化，确保符合平台爆款逻辑；\n"
        "- 心理专家：从认知心理学审查——好奇缺口(留悬念)、损失厌恶(强调代价)、"
        "情绪共振(代入感)、认知流畅(短句易理解)，优化文案的情绪价值与吸引力；\n"
        "- 视频导演：审查整体节奏是否流畅、时长分配、情绪起伏曲线，避免平铺直叙；\n"
        "- 视频字幕专家：审查旁白断句是否利于逐字字幕同步、是否适合竖屏展示；\n"
        "- 分镜设计专家：审查每个 scene 的画面冲击力(景别/机位/运镜/视觉张力)；\n"
        "- 视觉设计专家：审查整体视觉美感、色彩情绪、画面氛围是否高级有设计感；\n"
        "- 文案编辑：确保内容 100% 忠于用户原意，不编造事实数据，只做口语化+节奏优化。\n\n"
        "输出优化后的严格 JSON 数组，每个分镜字段：\n"
        '{"scene": 画面描述(带冲击力), "narration": 该镜旁白, "pause": 停顿节奏说明如"快-顿-慢", '
        '"emphasis": 需要加重语气凸显的重点词, "tone": 该镜情绪(如急切/郑重/反差/轻松), "seconds": 时长}。\n'
        "硬性要求：\n"
        "- 所有分镜 narration 拼接后须覆盖原脚本全部核心内容，不得丢失或添加用户没有的事实；\n"
        "- narration 用短句(8~20字)，可用……(停顿)、，(短停顿)、!(强调)体现停顿节奏；\n"
        "- pause/emphasis/tone 用于指导配音停顿、重点加重与情绪，不能改变 narration 原意；\n"
        "- 尽量将正文拆成多条'分条'式短分镜(每条一句一个要点，如'第一条…''第二条…')，"
        "层层递进，符合鲁大猿分条爆款结构；\n"
        "- 如现有分镜节奏平缓/冲击不足，必须大胆调整 scene 与 seconds，使其符合爆款节奏；\n"
        "- 只输出 JSON，不要任何解释或 markdown。"
    )
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


@tool
def tts(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> str:
    """文字转语音，生成 mp3 并返回文件路径。使用免费 edge-tts。"""
    import edge_tts
    out = os.path.join(config.OUTPUT_DIR, "voice.mp3")
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    # edge-tts 异步 API
    import asyncio
    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out)
    asyncio.run(_run())
    return out


def tts_with_subtitle(text: str, voice: str = "zh-CN-XiaoxiaoNeural",
                      rate: str = "+0%", out_stem: str = "voice") -> str:
    """文字转语音 + 生成逐句同步字幕(SRT)。利用 edge-tts 的 SentenceBoundary
    时间戳，保证字幕与语音严格对齐。返回 (mp3路径, srt路径)。
    rate: 语速，如 "+50%" 急促、"-20%" 舒缓。out_stem: 输出文件名前缀，
    避免多段配音(标题/内容)互相覆盖。"""
    import edge_tts
    import asyncio
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    mp3 = os.path.join(config.OUTPUT_DIR, f"{out_stem}.mp3")
    srt = os.path.join(config.OUTPUT_DIR, f"{out_stem}.srt")
    # 每次调用清空旧文件，避免多次 tts 追加污染
    for p in (mp3, srt):
        if os.path.exists(p):
            os.remove(p)
    submaker = edge_tts.SubMaker()
    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                with open(mp3, "ab") as f:
                    f.write(chunk["data"])
            elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                submaker.feed(chunk)
    asyncio.run(_run())
    with open(srt, "w", encoding="utf-8") as f:
        f.write(submaker.get_srt())
    return mp3, srt


def get_cn_font():
    """探测一个可用的中文字体文件（macOS）。"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def split_title_lines(title: str, max_len: int = 8) -> list:
    """把标题按语义分割成多行，避免超屏。优先按连接词/标点断行，
    否则按字数均分，保证每行不超过 max_len 字。"""
    if not title:
        return [""]
    # 先按连接词/符号切分
    seps = ["之", "：", "，", ",", "、", "|", "-", "—", "·", " "]
    for sep in seps:
        if sep in title:
            parts = [p for p in title.split(sep) if p]
            if parts:
                # 若切出的第一段很短(<=max_len)，用这些 parts 拼行
                lines = []
                cur = ""
                for p in parts:
                    if len(cur) + len(p) + 1 <= max_len and cur:
                        cur += sep + p
                    elif not cur:
                        cur = p
                    else:
                        lines.append(cur)
                        cur = p
                if cur:
                    lines.append(cur)
                if len(lines) > 1 or len(title) > max_len:
                    return lines if len(lines) > 1 else [title[:max_len], title[max_len:]]
    # 无连接词：按字数均分两行
    if len(title) <= max_len:
        return [title]
    half = (len(title) + 1) // 2
    return [title[:half], title[half:]]


def srt_to_ass(srt_path: str, out_path: str, title: str = "",
               play_w=1080, play_h=1920, title_hold: float = 4.5,
               content_fs: int = 74, title_fs: int = 110) -> str:
    """把 SRT 转成"逐字打字机"ASS 字幕 + 标题特写动画：
    - 标题：开头大字居中，金色描边，缩放强调动画（气氛烘托），持续 title_hold 秒后消失。
    - 内容：每条字幕随读音逐字变亮(karaoke)，读完一句整句消失，下句换高度位置。
    - content_fs/title_fs: 内容/标题字号(专家整改时通过此参数调整)。
    返回 ASS 路径。"""
    # 内容字幕固定到画面下方(竖屏1920高，y≈1500 下部偏中，留出上下安全区)，不再跳位
    content_y = 1500
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {play_w}\nPlayResY: {play_h}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Title, PingFang SC, 120, &H00A8E6FF, &H000000FF, &H005A2A00, &H96FFFFFF, -1, 0, 0, 0, 100, 100, 0, 0, 1, 10, 6, 5, 30, 30, 40, 1\n"
        "Style: Karaoke, PingFang SC, 60, &H0050B4FF, &H80101010, &H005A2A00, &H96FFFFFF, -1, 0, 0, 0, 100, 100, 0, 0, 1, 9, 6, 5, 30, 30, 40, 1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []

    # 1) 标题特写：0~title_hold 秒大字居中，按语义分割成多行、缩放强调 + 金色描边
    if title:
        # 语义分割成多行，避免超屏
        lines = split_title_lines(title)
        ts = min(title_fs, max(52, int(940 / max(len(lines), 1))))
        disp = r"\N".join(
            ln.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
            for ln in lines
        )
        def fmt_t(t):
            hh = int(t // 3600); mm = int((t % 3600) // 60)
            ss = int(t % 60); cc = int((t % 1) * 100)
            return f"{hh}:{mm:02d}:{ss:02d}.{cc:02d}"
        events.append(
            f"Dialogue: 0,0:00:00.00,{fmt_t(title_hold)},Title,,0,0,0,,"
            f"{{\\an5\\pos({play_w//2},{play_h//2-180})\\fad(150,350)\\t(0,350,"
            f"\\fs{ts}\\bord10)}}{disp}"
        )

    # 2) 内容字幕：逐字打字机
    idx = 0
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        time_line = None
        text_lines = []
        for l in lines:
            if "-->" in l:
                time_line = l
            elif re.fullmatch(r"\d+", l.strip()):  # SRT 序号行，跳过
                continue
            else:
                text_lines.append(l)
        if not time_line:
            continue
        def ts(s):
            parts = s.replace(",", ".").split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        m = re.match(r"(\d+:\d+:\d+[.,]\d+)\s*-->\s*(\d+:\d+:\d+[.,]\d+)", time_line)
        if not m:
            continue
        start, end = ts(m.group(1)), ts(m.group(2))
        text = " ".join(text_lines).strip()
        if not text:
            continue
        n = len(text)
        if n == 0:
            continue
        dur = end - start
        # 逐字 karaoke：每个字一个 \k，时长=句长/字数（厘秒）
        per_ck = max(int(dur / n * 100), 1)
        # 字号固定大(竖屏醒目)；超长句每行约 12 字处插入 \N 换行，逐字仍连续
        fs = content_fs
        chars_per_line = max(6, int(920 / fs))  # 每行约 12 字
        parts = []
        for i, c in enumerate(text):
            if i > 0 and i % chars_per_line == 0:
                parts.append(r"\N")
            parts.append(f"{{\\k{per_ck}}}{c}")
        karaoke = "".join(parts)
        y = content_y  # 固定下方
        idx += 1
        def fmt(t):
            hh = int(t // 3600); mm = int((t % 3600) // 60)
            ss = int(t % 60); cc = int((t % 1) * 100)
            return f"{hh}:{mm:02d}:{ss:02d}.{cc:02d}"
        events.append(
            f"Dialogue: 0,{fmt(start)},{fmt(end)},Karaoke,,0,0,0,,"
            f"{{\\an5\\pos({play_w//2},{y})\\fs{fs}\\fad(180,0)}}{karaoke}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return out_path


def get_audio_duration(path: str) -> float:
    """读取音频时长(秒)。"""
    ffmpeg = get_ffmpeg()
    r = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0


def concat_audio(audios: list, out_path: str) -> str:
    """顺序拼接多段音频(mp3/wav)，返回输出路径。用于"标题朗读 + 内容朗读"合一条配音。"""
    ffmpeg = get_ffmpeg()
    list_file = os.path.join(config.OUTPUT_DIR, "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for a in audios:
            f.write(f"file '{os.path.abspath(a)}'\n")
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c:a", "libmp3lame", "-q:a", "2", out_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"音频拼接失败: {res.stderr[-400:]}")
    return out_path


def offset_srt(srt_path: str, offset_sec: float, out_path: str) -> str:
    """把 SRT 所有字幕时间整体向后偏移 offset_sec 秒。用于内容字幕紧接标题朗读后。"""
    import datetime
    def add(t, o):
        dt = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second,
                                microseconds=t.microsecond) + datetime.timedelta(seconds=o)
        return dt
    lines_out = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            lines_out.append(block)
            continue
        g = m.groups()
        def mk(g):
            return f"{int(g[0]):02d}:{int(g[1]):02d}:{int(g[2]):02d},{int(g[3]):03d}"
        s0 = datetime.timedelta(hours=int(g[0]), minutes=int(g[1]), seconds=int(g[2]),
                                milliseconds=int(g[3])) + datetime.timedelta(seconds=offset_sec)
        e0 = datetime.timedelta(hours=int(g[4]), minutes=int(g[5]), seconds=int(g[6]),
                                milliseconds=int(g[7])) + datetime.timedelta(seconds=offset_sec)
        def fmt(d):
            hh, rem = divmod(int(d.total_seconds()), 3600)
            mm, ss = divmod(rem, 60)
            ms = d.microseconds // 1000
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
        lines[1] = f"{fmt(s0)} --> {fmt(e0)}"
        lines_out.append("\n".join(lines))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines_out) + "\n")
    return out_path


def make_bg_music(duration: float, out_path: str) -> str:
    """用 ffmpeg 合成一段柔和背景音乐(无版权)：几个低频正弦叠加 + 淡入淡出，
    营造氛围烘托。返回 wav 路径。"""
    ffmpeg = get_ffmpeg()
    dur = max(duration, 5)
    fade_out = max(1.0, dur - 3)
    expr = ("0.030*sin(2*PI*220*t)+0.022*sin(2*PI*277.18*t)"
            "+0.016*sin(2*PI*329.63*t)+0.012*sin(2*PI*392*t)")
    cmd = [
        ffmpeg, "-y", "-f", "lavfi", "-i",
        f"aevalsrc='{expr}':s=44100:d={dur:.1f}",
        "-af", f"afade=t=in:st=0:d=2.5,afade=t=out:st={fade_out:.1f}:d=3,"
                "lowpass=f=1400,volume=0.9",
        "-c:a", "pcm_s16le", out_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"背景音乐生成失败: {res.stderr[-400:]}")
    return out_path


@tool
def gen_image(prompt: str, out_name: str = "background.png") -> str:
    """文生图：用智谱 cogview-3-flash 生成竖屏 9:16 背景图（免费）。返回本地图片路径。
    out_name: 输出文件名，换背景时传不同名避免覆盖。"""
    if not config.IMAGE_API_KEY:
        raise RuntimeError("未配置智谱文生图 key（config.IMAGE_API_KEY）")
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("缺少 openai 依赖，请先 pip install openai")
    client = OpenAI(api_key=config.IMAGE_API_KEY, base_url=config.IMAGE_BASE_URL)
    resp = client.images.generate(model=config.IMAGE_MODEL,
                                  prompt=prompt, size="1024x1792")
    url = resp.data[0].url
    if not url:
        raise RuntimeError("文生图未返回图片 URL")
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    img_path = os.path.join(config.OUTPUT_DIR, out_name)
    import urllib.request
    urllib.request.urlretrieve(url, img_path)
    # 色彩增强兜底：即使模型生成偏灰/低饱和，也用 ffmpeg 提饱和、加对比，确保彩色生动
    try:
        _ff = get_ffmpeg()
        _tmp = img_path + ".tmp.png"
        _r = subprocess.run(
            [_ff, "-y", "-i", img_path,
             "-vf", "eq=saturation=1.8:contrast=1.15:brightness=0.03",
             "-q:v", "2", _tmp], capture_output=True, text=True)
        if _r.returncode == 0 and os.path.exists(_tmp):
            os.replace(_tmp, img_path)
    except Exception:
        pass
    return img_path


@tool
def extract_frames(video_path: str, times: str = "1,12,24", out_prefix: str = "frame") -> str:
    """从成片按秒抽帧，输出图片路径列表(逗号分隔)，供多模态专家审查真实画面质量。
    - video_path: 视频路径。
    - times: 逗号分隔的秒数(如 '1,12,24')。
    - out_prefix: 输出文件名前缀。
    返回逗号分隔的 PNG 路径列表。"""
    ffmpeg = get_ffmpeg()
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    secs = [t for t in times.split(",") if t]
    paths = []
    for s in secs:
        out = os.path.join(config.OUTPUT_DIR, f"{out_prefix}_{s}.png")
        # 抽帧并缩到 540 宽(竖屏540x960)，显著减小体积，便于多模态多图上传
        res = subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-vf",
             f"select='eq(t,{s})',scale=540:-2", "-vsync", "0",
             "-frames:v", "1", out],
            capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(out):
            paths.append(out)
    return ",".join(paths)


@tool
def gen_frame_review(topic: str, frame_paths: str, review_type: str = "round1") -> str:
    """让抖音爆款视频专家、视频导演、视频设计师三位专家'亲眼'审查成片的真实抽帧，
    从画面美感、背景质感、字幕可读性、字号大小、标题冲击力、整体设计感等维度，
    输出结构化的整改建议 JSON。

    - topic: 视频标题。
    - frame_paths: 逗号分隔的抽帧 PNG 路径。
    - review_type: round1(第一轮整改，给大刀阔斧建议) / round2(第二轮微调)。
    返回 JSON：{ "background_prompt": 若需换背景给出唯美背景新提示词或空,
                 "brightness": 建议背景亮度调整(-1~1, 负变暗正变亮),
                 "saturation": 建议饱和度, "font_size": 建议内容字号,
                 "title_size": 建议标题字号, "bg_music_style": 建议背景音乐情绪,
                 "overall_notes": 一句话总结 }
    用智谱免费多模态 glm-4v-flash 真实看图。"""
    import base64 as _b64
    from openai import OpenAI
    client = OpenAI(api_key=config.VISION_API_KEY, base_url=config.VISION_BASE_URL)
    content = []
    for p in frame_paths.split(","):
        p = p.strip()
        if not p or not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            b64 = _b64.b64encode(f.read()).decode()
        mime = "image/png"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    role_hint = ("第一轮：这是打造抖音爆款的关键整改，必须严苛、专业、一针见血。"
                 "背景是否足够唯美彩色、字幕字号与排版是否醒目高级、标题是否有爆款冲击力、"
                 "整体设计感与美感是否达到平台头部水平。")
    if review_type == "round2":
        role_hint = ("第二轮微调：上一轮问题是否已改善？仍有哪些细节影响爆款观感必须改？"
                     "只提高价值硬性整改点，其余保持。")
    content.append({
        "type": "text",
        "text": (
            "你是由【抖音爆款视频专家】【视频导演】【视觉设计专家】【视频字幕专家】四名权威专家"
            "组成的爆款质检委员会。请'亲眼'逐帧审查这些真实成片抽帧，以打造抖音爆款为最高标准。\n"
            "【对标爆款标杆：抖音知识科普头部账号@鲁大猿的画面与制作水准】"
            "鲁大猿的视频以硬核知识科普爆款著称：画面高密度信息、视觉冲击力强、"
            "配色高级有设计感、字幕醒目专业、节奏紧凑有推进感、情绪有'恍然大悟'的爽感。"
            "请对标这一头部水准严格审查。\n"
            f"视频标题《{topic}》。{role_hint}\n"
            "每位专家必须从自己专业角度对实际画面提出至少 1 条硬性可执行整改意见，绝不允许空泛好评：\n"
            "- 视觉设计专家：审查色彩搭配、构图、层次、留白、画面冲击力，背景是否唯美彩色高级；\n"
            "- 视频导演：审查节奏、换场是否柔和连贯、情绪表达、镜头语言；\n"
            "- 视频字幕专家：审查字幕字号、位置、可读性、排版设计感、是否与画面协调；\n"
            "- 爆款专家：审查是否符合抖音爆款观感、是否抓人、是否值得点赞关注。\n"
            "输出严格 JSON(不要解释、不要markdown)：\n"
            '{"background_prompt": 换背景的唯美彩色新提示词或"" , '
            '"brightness": 亮度数字(-1~1), "saturation": 饱和度数字(0~2), '
            '"font_size": 建议内容字号(50~130), "title_size": 建议标题字号(60~170), '
            '"title_speed": 标题朗读语速(建议值,如-20~+20), '
            '"scene_transition": 换场方式建议(柔和/淡入淡出/留白,文字描述), '
            '"bg_music_style": 背景音乐情绪(如:轻松/激昂/温暖/空灵), '
            '"design_notes": 视觉/字幕/导演的综合硬性整改意见(2-4句话,必须具体专业), '
            '"overall_notes": 一句话总结是否达标爆款}。'
        ),
    })
    resp = client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


@tool
def gen_video(prompt: str) -> str:
    """文生视频：生成竖屏 9:16 素材。此为你需要接入的 API 占位。"""
    # TODO: 换成可灵/即梦/通义万相 的文生视频 API（异步任务 + 轮询结果）
    raise NotImplementedError("请接入你的文生视频 API（见 tools.py 内注释）")


@tool
def merge_video(scene_videos: str = "", voice_mp3: str = "", title: str = "",
                subtitle_ass: str = "", bg_music: str = "", background: str = "",
                title_end: float = 3.0, brightness: float = -0.30,
                saturation: float = 0.85) -> str:
    """合成最终成片。
    - scene_videos: 逗号分隔的视频路径；为空用背景图或纯色打底。
    - voice_mp3: 配音文件。
    - subtitle_ass: ASS 字幕文件(含标题特写动画 + 逐字打字机字幕)。
    - bg_music: 背景音乐 wav 路径，与配音混音(配音为主、背景低音量烘托)。
    - background: 背景图片路径；渲染时缩放填满 + 暗化压暗，突出文字。
    - title_end: 标题朗读结束秒数；此处淡出到黑场再淡入，实现换场效果。
    - brightness: 背景亮度调整(-1~1，默认-0.30，专家可据此整改)。
    - saturation: 背景饱和度(默认0.85，专家可据此整改)。
    """
    ffmpeg = get_ffmpeg()
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "final.mp4")

    vlist = [v.strip() for v in scene_videos.split(",") if v.strip()]
    if vlist:
        list_file = os.path.join(config.OUTPUT_DIR, "list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for v in vlist:
                f.write(f"file '{v}'\n")
        base = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file]
        bg_filter = "[0:v]null[bgv]"
    elif background and os.path.exists(background):
        # 背景图：缩放填满 + 可调亮度/饱和度(专家整改时用变量覆盖)
        base = [ffmpeg, "-y", "-loop", "1", "-i", background]
        bg_filter = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,eq=brightness={brightness}:saturation={saturation}[bgv]"
        )
    else:
        base = [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x1a1a2e:s=1080x1920:d=60"]
        bg_filter = "[0:v]null[bgv]"

    cmd = base + ["-i", voice_mp3]
    if bg_music:
        cmd += ["-i", bg_music]

    filter_parts = [bg_filter]

    # 字幕：标题特写 + 逐字打字机（都在 ASS 里）
    if subtitle_ass:
        abspath = os.path.abspath(subtitle_ass).replace("\\", "/")
        abspath = abspath.replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(f"[bgv]subtitles='{abspath}'[subv]")
        prev = "subv"
    else:
        filter_parts.append("[bgv]null[subv]")
        prev = "subv"

    # 换场：标题朗读结束后，柔和过渡到内容（不硬切黑场）。
    # 标题事件自带 \fad(150,350) 淡出，内容字幕首条 \fad 淡入，背景保持连贯，换场自然柔和。
    filter_parts.append(f"[{prev}]null[v]")

    # 音频混音：配音为主 + 背景低音量
    if bg_music:
        filter_parts.append("[1:a]volume=1.0[a0]")
        filter_parts.append("[2:a]volume=0.28[a1]")
        filter_parts.append("[a0][a1]amix=inputs=2:duration=shortest:normalize=0[aout]")
        audio_map = "[aout]"
    else:
        audio_map = "1:a"

    fc = ";".join(filter_parts)
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", audio_map]

    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart", out]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 合成失败: {res.stderr[-800:]}")
    return out
