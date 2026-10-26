"""本地网页界面：填标题+内容，一键生成抖音视频，实时显示执行流程。
启动：python app.py   -> 浏览器打开 http://127.0.0.1:5001
"""
import json
import os

from flask import Flask, request, Response, send_file, render_template

from make_video import create_video_stream

app = Flask(__name__)

OUT = os.path.abspath("output")
os.makedirs(OUT, exist_ok=True)


@app.route("/")
def index():
    resp = render_template("index.html")
    # 加防缓存头，避免浏览器加载旧页面导致 JS 报错
    from flask import make_response
    r = make_response(resp)
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    return r


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    content = (data.get("content") or "").strip()
    if not topic:
        return Response("data: " + json.dumps({"msg": "❌ 请填写标题/主题"}) + "\n\n",
                        mimetype="text/event-stream")

    def stream():
        try:
            gen = create_video_stream(topic, content)
            for msg in gen:
                if msg.startswith("output/"):  # 最后一个是成片路径
                    yield "data: " + json.dumps({"video": f"/video?f={os.path.basename(msg)}"}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"msg": msg}) + "\n\n"
        except Exception as e:  # noqa
            yield "data: " + json.dumps({"msg": f"❌ 生成失败：{e}"}) + "\n\n"

    return Response(stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/video")
def video():
    f = request.args.get("f")
    if not f:
        return "missing", 400
    return send_file(os.path.join(OUT, f))


if __name__ == "__main__":
    # 注意：5000 端口常被 macOS 系统服务(ControlCenter)占用，改用 5001
    print("启动成功！浏览器打开: http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
