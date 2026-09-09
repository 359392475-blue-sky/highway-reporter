# 高速违法举报助手

从行车记录仪视频中筛出疑似事件、提取证据帧并生成可人工核对的材料。

源码包含本地车辆检测、事件追踪、云端视觉读取和证据材料确认页面。识别结果需要人工复核；本项目不会替你自动提交举报，也不保证任何识别准确率。

## 运行

需要 Python 3.9+、FFmpeg。建议在虚拟环境安装依赖：

~~~sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python full_pipeline.py --check --dry-run
python full_pipeline.py /path/to/video.mp4 --dry-run
~~~

完整云端读取需要在 .env 中填写自己的 VOLCENGINE_API_KEY。模型缺失时，按 Ultralytics 官方说明取得 yolov8n.pt 并放入 data/。公开包不包含个人视频、车牌证据、模型密钥或运行结果。

带材料确认界面的处理：python full_pipeline.py /path/to/video.mp4 -o output/。

## 下载

GitHub Releases 提供源码测试包和安装说明；这是 Python 测试包，不是已签名的 macOS 安装器。

## 许可

本公开源码按 GNU AGPL v3 提供，许可证全文见 LICENSE。项目使用 Ultralytics YOLO；其代码、模型及其他依赖仍须遵循各自的许可证和使用条件。
