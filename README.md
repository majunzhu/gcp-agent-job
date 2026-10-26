# douyin_agent — 用 LangChain 自建抖音视频制作智能体

不依赖扣子/积分，纯代码跑通「主题 → 脚本 → 分镜 → 配音 → 合成」链路。

## 目录结构
```
douyin_agent/
├── config.py      # 模型 / key / 输出目录配置（改这里）
├── tools.py       # 工具层：每个环节一个 @tool
├── agent.py       # 编排层：LangChain ReAct Agent
├── main.py        # 入口
└── output/        # 生成产物（voice.mp3 / final.mp4）
```

## 运行步骤

1. 装依赖
   ```bash
   pip install -r requirements.txt
   ```
   另需本机安装 [ffmpeg](https://ffmpeg.org/)（`brew install ffmpeg`）。

2. 改 `config.py`，填入你的 LLM key：
   - DeepSeek：`LLM_BASE_URL=https://api.deepseek.com/v1`，`LLM_MODEL=deepseek-chat`
   - 通义千问 / 豆包Seed 等同理换 base_url + model

3. 先跑通不依赖外部画面 API 的链路：
   ```bash
   python main.py "AI 如何帮我写周报"
   ```
   会看到 Agent 一步步调用工具：生成脚本 → 拆分镜 → 配音 → （画面/合成处会提示待接入）。

4. 接入画面 API（可选）：编辑 `tools.py` 里 `gen_image` / `gen_video` 两个函数的 TODO 占位，
   按你开通的平台（即梦/可灵/通义万相）填真实调用。

## 关键点
- `agent.py` 的 `verbose=True` 可观察 Agent 的 ReAct 推理过程，对你面试讲 Agent 也有帮助。
- 画面合成是整条链路上唯一强依赖外部 API 的环节；脚本/分镜/配音/合成可完全本地跑通。
