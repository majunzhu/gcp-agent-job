"""抖音视频生成流水线：
命令行：python make_video.py "主题" ["额外内容"]
网页版：from make_video import create_video; create_video("主题", "内容")
流程：主题 -> 脚本 -> 分镜 -> 配音 -> 合成成片（无画面素材时背景画面打底）。
"""
import json
import os
import sys
import time

from langchain_openai.chat_models.base import OpenAIRateLimitError

import config
from tools import (
    gen_script, gen_storyboard, tts_with_subtitle, srt_to_ass, merge_video,
    make_bg_music, get_audio_duration, concat_audio, offset_srt, gen_image,
    gen_review, extract_frames, gen_frame_review,
)


def retry_call(fn, *args, retries=6, wait=10, **kwargs):
    """调用工具，遇到限流(429)自动等待后重试。免费模型高峰易限流，多等几次。"""
    last_exc = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except OpenAIRateLimitError as e:
            last_exc = e
            print(f"  限流，{wait}s 后重试 ({i+1}/{retries})...")
            time.sleep(wait)
    raise RuntimeError(f"多次重试仍被限流，请稍后再试（{last_exc}）")


def safe_loads_json(raw: str):
    """健壮解析 LLM 返回的 JSON：先直接解析，失败则提取首个 [...] 或 {...} 再解析。"""
    raw = raw.strip()
    # 去掉可能的 ```json 围栏
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 提取首个 '[' 到其匹配 ']' 的数组片段
    start = raw.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except Exception:
                        pass
    # 提取首个 '{' 到匹配 '}'
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except Exception:
                        pass
    raise ValueError(f"无法解析 LLM 返回的 JSON: {raw[:200]}")


def create_video_stream(topic: str, content: str = "") -> "generator":
    """流式出片流程：每一步 yield 一段进度文本，最后 yield 成片路径。
    供网页(SSE)和命令行共用，实现实时执行反馈。"""
    # 0) 根据标题+内容分析生成竖屏唯美背景图（明亮通透，突出文字又不深暗）
    yield "[1/6] 🖼️ 正在生成竖屏唯美背景图..."
    try:
        bg_prompt = (
            f"为短视频设计一张唯美明亮、色彩极其丰富的竖屏背景图，主题：{topic}。{content}。"
            "【最高优先级硬性要求：必须是高饱和、鲜艳、绚丽的多彩画面，绝不能是黑白、"
            "灰色、单色或低饱和！】"
            "场景指定：蔚蓝天空+金色阳光+翠绿草地+彩色花海(红黄粉紫)+暖橙晚霞等，"
            "色彩鲜明对比强，积极向上有活力、高级治愈感；"
            "画面留出大量干净留白给文字，构图居中，质感细腻通透，符合抖音高级爆款审美。"
        )
        background = gen_image.invoke(bg_prompt)
        yield f"[1/6] ✅ 唯美背景图已生成：{background}"
    except Exception as e:
        background = ""
        yield f"[1/6] ⚠️ 背景图生成失败（{str(e)[:40]}），用纯色背景兜底"

    # 1) 生成口播脚本
    yield f"[2/6] 🧠 正在用 AI 生成口播脚本（主题：{topic}）..."
    script = retry_call(gen_script.invoke, {"topic": topic, "content": content})
    yield f"[2/6] ✅ 脚本生成完成：{script[:60]}..."

    # 2) 拆分成镜
    yield "[3/6] 📋 正在把脚本拆分成镜..."
    raw = retry_call(gen_storyboard.invoke, script)
    shots = safe_loads_json(raw)
    yield f"[3/6] ✅ 分镜完成，共 {len(shots)} 个分镜"

    # 2.5) 爆款级质量把控：抖音爆款专家+心理专家+导演+分镜+字幕+文案 六角色审查
    yield "[4/6] 🎬 爆款质量把控（爆款专家+心理专家+导演+分镜+字幕+文案 六角色审查优化）..."
    raw_rev = retry_call(gen_review.invoke, {"topic": topic, "script": script, "shots_json": raw})
    shots = safe_loads_json(raw_rev)
    yield f"[4/6] ✅ 专家审查优化完成，分镜调整为 {len(shots)} 个"

    # 3) 配音：标题有节奏朗读 + 内容正常朗读，拼接；字幕时间偏移
    yield "[5/6] 🔊 正在生成配音（标题有节奏 + 内容）+ 同步字幕..."
    # 标题朗读：慢而清晰、有节奏(抖音爆款标题风格：字字有力)，用独立文件避免覆盖
    import re as _re
    _title_txt = _re.sub(r"之", "，", topic)  # 在"之"处加停顿，读起来更有节奏
    title_voice, title_srt = tts_with_subtitle(_title_txt, rate="-10%", out_stem="title_voice")
    dur_title = get_audio_duration(title_voice)
    # 内容朗读：正常语速；去掉分镜序号(1、2、3...)保证语音与字幕严格对齐
    narration = ""
    for s in shots:
        _n = s.get("narration", "")
        _n = _re.sub(r"^\s*\d+[、.．]?\s*", "", _n)  # 去每条开头的序号
        narration += _n
    content_voice, content_srt = tts_with_subtitle(narration, rate="+5%", out_stem="content_voice")
    # 拼接音频：标题在前，内容在后
    voice = concat_audio([title_voice, content_voice],
                         os.path.join(config.OUTPUT_DIR, "voice_full.mp3"))
    dur_voice = get_audio_duration(voice)
    # 内容字幕整体后移标题时长；标题字幕保持从 0 开始
    content_srt_off = offset_srt(content_srt, dur_title,
                                 os.path.join(config.OUTPUT_DIR, "content_srt_off.srt"))
    yield f"[5/6] ✅ 配音+字幕完成（标题{dur_title:.1f}s + 内容{dur_voice-dur_title:.1f}s）"

    # 4) 合成：唯美背景 + 标题特写(多行) + 换场 + 逐字字幕 + 背景音乐
    def _assemble(background, brightness, saturation, content_fs, title_fs):
        merged_srt = content_srt_off
        ass = srt_to_ass(merged_srt, os.path.join(config.OUTPUT_DIR, "subtitle.ass"),
                         title=topic, title_hold=dur_title + 0.4,
                         content_fs=content_fs, title_fs=title_fs)
        bg = make_bg_music(dur_voice, os.path.join(config.OUTPUT_DIR, "bg_music.wav"))
        return merge_video.invoke({
            "scene_videos": "", "voice_mp3": voice, "title": topic,
            "subtitle_ass": ass, "bg_music": bg, "background": background,
            "title_end": dur_title, "brightness": brightness, "saturation": saturation,
        })

    # 初始参数（唯美：背景稍暗化保证文字可读，但保留通透感）
    brightness = -0.22
    saturation = 0.95
    content_fs = 74
    title_fs = 110
    curr_bg = background

    # [6/8] 第一版合成 + 抽帧供专家看真实画面
    yield f"[6/8] 🎞️ 合成第一版成片（唯美背景+标题特写+换场+逐字字幕+音乐）..."
    v1 = _assemble(curr_bg, brightness, saturation, content_fs, title_fs)
    frames1 = extract_frames.invoke({"video_path": v1, "times": "1.2,12,24",
                                     "out_prefix": "q1"})
    yield f"[6/8] ✅ 第一版已合成，抽帧 {frames1}"

    # [7/8] 第1轮整改：爆款专家+导演+设计师亲眼看真实画面
    yield "[7/8] 🎬 专家亲眼看真实成片，第1轮整改（爆款专家+视频导演+视频设计师）..."
    try:
        rev1_raw = retry_call(gen_frame_review.invoke,
                              {"topic": topic, "frame_paths": frames1, "review_type": "round1"})
        rev1 = safe_loads_json(rev1_raw)
        # 应用整改：换唯美背景 / 调亮度饱和度 / 调字号
        if rev1.get("background_prompt"):
            try:
                yield "      ↳ 专家建议换更唯美背景，正在重新生成..."
                bg2 = f"background_r1.png"
                curr_bg = gen_image.invoke({"prompt": rev1["background_prompt"], "out_name": bg2})
            except Exception as e:
                yield f"      ↳ 换背景失败({str(e)[:30]})，沿用当前背景"
        brightness = float(rev1.get("brightness", brightness))
        saturation = float(rev1.get("saturation", saturation))
        content_fs = int(rev1.get("font_size", content_fs))
        title_fs = int(rev1.get("title_size", title_fs))
        yield f"      ↳ 整改意见：背景亮度{brightness}/饱和度{saturation}/内容字号{content_fs}/标题字号{title_fs}。{rev1.get('overall_notes','')}"
        # 用整改参数重新合成
        v2 = _assemble(curr_bg, brightness, saturation, content_fs, title_fs)
        yield "[7/8] ✅ 第1轮整改完成，已重新合成"
    except Exception as e:
        v2 = v1
        yield f"[7/8] ⚠️ 第1轮专家整改失败({str(e)[:40]})，沿用第一版"

    # [8/8] 第2轮整改：专家再看真实画面微调
    yield "[8/8] 🎬 专家再次亲眼看真实成片，第2轮微调（保证爆款质量）..."
    try:
        frames2 = extract_frames.invoke({"video_path": v2, "times": "1.2,12,24",
                                         "out_prefix": "q2"})
        rev2_raw = retry_call(gen_frame_review.invoke,
                              {"topic": topic, "frame_paths": frames2, "review_type": "round2"})
        rev2 = safe_loads_json(rev2_raw)
        if rev2.get("background_prompt"):
            try:
                yield "      ↳ 第2轮仍建议换背景，重新生成唯美背景..."
                curr_bg = gen_image.invoke({"prompt": rev2["background_prompt"], "out_name": "background_r2.png"})
            except Exception:
                pass
        brightness = float(rev2.get("brightness", brightness))
        saturation = float(rev2.get("saturation", saturation))
        content_fs = int(rev2.get("font_size", content_fs))
        title_fs = int(rev2.get("title_size", title_fs))
        yield f"      ↳ 第2轮意见：{rev2.get('overall_notes','')}"
        video = _assemble(curr_bg, brightness, saturation, content_fs, title_fs)
        yield "[8/8] ✅ 第2轮微调完成，成片定稿"
    except Exception as e:
        video = v2
        yield f"[8/8] ⚠️ 第2轮微调失败({str(e)[:40]})，采用上一版定稿"

    yield f"✅ 最终成片已生成：{video}"
    yield video  # 最后一个 yield 返回成片路径


def create_video(topic: str, content: str = "") -> str:
    """完整出片流程，返回成片路径。content 为可选补充素材/要求。"""
    last = None
    for msg in create_video_stream(topic, content):
        print(msg)
        last = msg
    return last  # 最后一个 yield 是视频路径


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI 如何帮我写周报"
    content = sys.argv[2] if len(sys.argv) > 2 else ""
    create_video(topic, content)


if __name__ == "__main__":
    main()
