# 部署指南

当前后端运行五模型平均集成的 `ResNet-18 + TabPFN + Transformer` 二分类方案（各成员权重 20%，训练 Dropout 为 0.1）。Git 仓库仅包含源码与公共配置，不包含模型权重、病例数据、连接口令和安装包。

---

## 一、前端开发与 Android 打包

### 1. 本地网页调试（连接远程服务）

环境要求：Git、Node.js 24。

```powershell
git clone --branch master https://github.com/Linexox/OPMD-Pred.git
cd OPMD-Pred/apps/mobile
npm ci
```

在 `apps/mobile/.env.local` 中配置后端地址与配对口令（该文件已被 Git 忽略）：

```dotenv
VITE_DEMO_SERVICE_URL=http://www.ming56.cn:9020
VITE_DEMO_PAIRING_TOKEN=<配对口令>
```

启动开发服务与构建检查：

```powershell
npm run dev     # 访问 http://127.0.0.1:5173
npm run build   # 类型检查与前端构建
```

说明：
- 修改 `.env.local` 后需重启 `npm run dev`。客户端已保存的连接配置可在页面“我的 → 连接设置”中覆盖。
- 开发网页版访问与 `.env.local` 一致的后端地址时，API和照片请求由本地Vite服务转发，避免浏览器直接连接公网时受到本机代理影响。APK和正式网页构建仍使用填写的服务地址。
- 前端构建会将 `.env.local` 中的口令打包进静态资源，内置口令的构建产物仅限内部测试使用。

### 2. Android APK 打包

- 现有安装包：`../交付成果/APP/口腔随访-1.3.9.apk`（线上下载：`http://www.ming56.cn:9020/download`）。仅开发前端界面无需配置 Android 环境。
- 自行打包环境要求：Node.js 24、Java 21、Android SDK（API 36 及对应 Build-Tools）。

打包步骤：
1. 在仓库同级目录创建 `../运行数据/pairing-token.txt`，写入目标后端的配对口令。
2. 注意：`apps/mobile/build-android.ps1` 硬编码了 `D:\.APP\nodejs24\npm.cmd` 和 `npx.cmd` 路径。若本机 Node.js 安装在其他位置，运行前需先将该脚本中的固定路径改为 `npm` / `npx`。
3. 在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File apps/mobile/build-android.ps1 `
  -Sdk "D:/path/to/android-sdk" `
  -Java "D:/path/to/java21"
```

可选参数：
- `-ServiceUrl <URL>`：指定默认后端地址（未传时默认为 `http://www.ming56.cn:9020`）。
- `-NoDefaultConnection`：生成不预填服务地址和口令的安装包（无需准备 `pairing-token.txt`）。
- `-SkipInstall`：跳过 `npm ci`。

构建输出位于 `apps/mobile/android/app/build/outputs/apk/debug/app-debug.apk`，并自动同步至 `../运行数据/distribution/app-debug.apk`。

---

## 二、后端服务独立部署（Linux CPU）

以下命令均在 Linux 仓库根目录下执行。参考运行环境：Python 3.14.4、PyTorch 2.14.0+cpu、TabPFN 9.2.0。

### 1. 放置模型包

将私有模型目录 `models/app/binary-mean/`（含 `manifest.json`、共享编码器及 `expert0`～`expert4` 共 43 个文件）完整放入仓库对应路径。模型包含训练参考上下文，仅限服务端存放，切勿提交至 Git 或打入客户端安装包。

```text
OPMD-Pred/
├─ apps/server/binary_linux_service.py
├─ models/app/binary-mean/
│  ├─ manifest.json
│  ├─ encoder.pt
│  ├─ tabpfn-v2-classifier-finetuned-zk73skhh.ckpt
│  └─ expert0/ … expert4/
└─ runtime/                 # 运行时生成：数据库、附件、口令与缓存
```

### 2. 安装 Python 依赖

`apps/server/requirements-binary-cpu.txt` 为当前二分类服务依赖清单（`requirements.txt` 与 `requirements.lock` 仅用于旧版三分类服务）：

```bash
mkdir -p runtime/cache/pip
export PIP_CACHE_DIR="$PWD/runtime/cache/pip"

python3.14 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0+cpu torchvision==0.29.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r apps/server/requirements-binary-cpu.txt
.venv/bin/python -m pip check
```

### 3. 生成配对口令与启动服务

生成配对口令 `runtime/pairing-token.txt` 并启动服务：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import secrets
p = Path('runtime/pairing-token.txt')
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    p.write_text(secrets.token_urlsafe(24), encoding='utf-8')
    p.chmod(0o600)
PY

.venv/bin/python apps/server/binary_linux_service.py
```

- 模型清单：默认加载 `models/app/binary-mean/manifest.json`；若存在 `runtime/binary-model-manifest.txt`，则按该文件指定的相对路径加载。
- 监听地址：默认监听 `127.0.0.1:8765`。运行数据（`records.sqlite3` 与上传附件）保存在 `runtime/`。

### 4. 健康检查

在另一终端执行：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import httpx
token = Path('runtime/pairing-token.txt').read_text().strip()
r = httpx.get('http://127.0.0.1:8765/api/health',
              headers={'Authorization': 'Bearer ' + token}, timeout=30)
r.raise_for_status()
print(r.json())
PY
```

正常返回 JSON 中 `model_id` 应为 `sysu-resnet18-tabpfn-transformer-mean5-seed42-20261105`。

### 5. 网络访问与 Systemd 后台常驻

- **本地开发转发**：服务仅绑定本地回环地址，远程开发机可通过 SSH 端口转发连接：
  ```bash
  ssh -N -L 8765:127.0.0.1:8765 <user>@<linux-host>
  ```
  随后将前端服务地址设为 `http://127.0.0.1:8765`。
- **公网暴露**：需通过反向代理或内网穿透（如 NPC/NPS）转发至 `127.0.0.1:8765`。当前团队公网入口为 `http://www.ming56.cn:9020`（注：9018 为 SSH 端口，9024 为 NPC 控制端口）。
- **Systemd 常驻**：
  1. 写入运行模式：`echo -n "binary" > runtime/model-mode.txt`（`launch-linux.sh` 依赖此文件切换至二分类入口，否则默认启动旧三分类服务）。
  2. 将 `apps/server/opmd-pred.service` 中的 `WorkingDirectory` 和 `ExecStart` 修改为实际仓库路径。
  3. 安装并启动用户级服务：
     ```bash
     mkdir -p ~/.config/systemd/user
     cp apps/server/opmd-pred.service ~/.config/systemd/user/
     systemctl --user daemon-reload
     systemctl --user enable --now opmd-pred.service
     ```

---

## 三、核心代码入口

| 模块 | 路径 | 说明 |
| :--- | :--- | :--- |
| 前端界面与接口 | `apps/mobile/src/App.vue`、`apps/mobile/src/api.ts` | 请求携带 `Authorization: Bearer <token>`，写入需带请求标识，分析请求声明支持二分类格式 |
| 后端路由与存储 | `apps/server/server.py`、`apps/server/core.py` | FastAPI 路由、SQLite 记录与附件管理 |
| 二分类推理接入 | `apps/server/binary_inputs.py`、`apps/server/binary_inference.py` | 输入特征整理与结果封装 |
| 冻结推理实现 | `apps/server/binary_runtime/` | 线上二分类与五模型集成推理代码，独立于研究代码 |
| 研究与训练脚本 | `scripts/research/`、`src/opmd_pred/` | 离线训练代码，修改不会自动作用于已部署服务 |
| Windows 旧开发服务 | `apps/server/launch.ps1` | 仅启动旧版三分类服务，不包含线上二分类运行环境 |

