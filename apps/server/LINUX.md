# Linux部署入口

更新：2026年11月5日。当前服务使用五个完整二分类模型，各占20%汇总概率；模型目录为 `models/app/binary-mean/`。

安装依赖、领取模型、准备口令、启动和检查的最小步骤，统一维护在[组员最小部署指南](../../DEPLOYMENT.md)。当前依赖清单是 `requirements-binary-cpu.txt`。

直接启动：

```bash
.venv/bin/python apps/server/binary_linux_service.py
```

该入口监听 `127.0.0.1:8765`。`launch-linux.sh`供兼容启动和systemd使用；其 `runtime/model-mode.txt` 必须为 `binary`，否则仍启动旧三分类服务。`models.example.json`、`requirements.txt`和`requirements.lock`是旧三分类入口的配置，不是当前二分类的部署步骤。
