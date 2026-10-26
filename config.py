"""集中配置：所有 key / 模型 / 接口都在这改。"""
import os

# ---------- 大语言模型（OpenAI 兼容接口）----------
# 当前使用智谱 GLM-4.7-Flash 免费模型
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "a8c3507dc3ba47e0af0cdc0b11ea1d2c.quya0hTFevtb00fJ")
LLM_MODEL    = os.getenv("LLM_MODEL",    "glm-4.7-flash")

# ---------- 画面/视频生成（按你开通的平台填，可先留空跑通链路）----------
# 智谱 cogview-3-flash：免费文生图，用同一个 LLM_API_KEY。
IMAGE_API_KEY  = os.getenv("IMAGE_API_KEY",  LLM_API_KEY)
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
IMAGE_MODEL    = os.getenv("IMAGE_MODEL",    "cogview-3-flash")
# 智谱 glm-4v-flash：免费多模态视觉模型，用于"专家看真实抽帧"质量审查
VISION_MODEL   = os.getenv("VISION_MODEL",   "glm-4v-flash")
VISION_API_KEY = os.getenv("VISION_API_KEY", LLM_API_KEY)
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
VIDEO_API_KEY  = os.getenv("VIDEO_API_KEY",  "")

# ---------- 输出目录 ----------
OUTPUT_DIR = "output"
