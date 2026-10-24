# Linux运行APP服务

部署目录示例：`/home/nexox/workspace/OPMD_Pred`。使用当前APP的两份模型，不需要训练数据。

先将`models/app/classifier.pt`、`image_encoder.pt`及模型清单放到对应目录；将现有APP口令私下保存为`runtime/pairing-token.txt`。模型和口令不会由Git克隆得到。`apps/server/models.local.json`可复制`models.example.json`，CPU部署保持`device`为`cpu`。

在项目`.venv`安装`requirements.txt`及包初始化需要的`loguru`。PyTorch和torchvision应使用相互对应的Linux版本；CPU安装示例：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0+cpu torchvision==0.29.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r apps/server/requirements.txt loguru
.venv/bin/python -m pip freeze > requirements-linux.lock
```

启动：

```bash
bash apps/server/launch-linux.sh
```

默认只监听`127.0.0.1:8765`，适合在同一台PC用NPC转发；口令不会打印。记录、附件、日志相关运行目录及缓存统一位于项目内`runtime/`。直接局域网访问需要显式增加`--host 0.0.0.0`。

需要后台持续运行时，在`~/workspace/OPMD_Pred`布局下安装用户服务：

```bash
mkdir -p ~/.config/systemd/user
cp apps/server/opmd-pred.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now opmd-pred.service
```

常用操作：

```bash
systemctl --user status opmd-pred.service --no-pager
systemctl --user restart opmd-pred.service
systemctl --user stop opmd-pred.service
journalctl --user -u opmd-pred.service -n 30 --no-pager
```

用户服务退出登录后是否继续、是否无需登录即可开机启动，取决于该账户的linger设置；部署记录注明实际核对结果。

APP公网隧道使用NPS服务端9020端口，内网目标设为此PC的`127.0.0.1:8765`。9018是既有SSH入口，9024是NPC通信端口，均保持原设置。
