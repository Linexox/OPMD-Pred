# 组员最小部署指南

核对日期：2026年11月5日。代码主线是 `master`；当前APP服务使用五个完整的 ResNet-18＋TabPFN＋Transformer 二分类模型，各占20%汇总预测，训练 Dropout 为0.1。

**只开发手机界面，按第一部分操作即可。** 独立运行模型服务再看第二部分。Git包含源码和共享配置，不包含模型、病例、口令和安装包。

## 一、前端开发：连接现有服务

准备 Git、Node.js 24。在D盘的工作目录打开PowerShell：

```powershell
git clone --branch master https://github.com/Linexox/OPMD-Pred.git
cd OPMD-Pred/apps/mobile
$env:npm_config_cache = Join-Path (Get-Location).Path '../../runtime/cache/npm'
npm ci
```

在 `apps/mobile/.env.local` 写入以下两行，把占位文字换成项目负责人私下提供的连接口令：

```dotenv
VITE_DEMO_SERVICE_URL=http://www.ming56.cn:9020
VITE_DEMO_PAIRING_TOKEN=在这里填入私下领取的口令
```

```powershell
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。修改 `.env.local` 后重启开发命令；已有连接设置会保留，需要时在APP“我的→连接设置”中修改。连接失败时先确认口令和公网服务是否可用。

页面构建检查：`npm run build`。`.env.local` 已被Git忽略；前端构建会将所填口令写进页面资源，所以这种连接方式仅供组内使用。现有服务会保存提交的就诊记录，联调请使用匿名测试资料。

### 需要Android安装包时

现有组内包为1.3.8，电脑上的成品位于仓库上级 `交付成果/APP/口腔随访-1.3.8.apk`；下载入口为 `http://www.ming56.cn:9020/download`。组员仅修改网页时，不必安装Android构建环境。

要自行打包，准备 Java 21、Android SDK（Android API 36及工程要求的构建工具）、Node.js 24；在仓库上级新建 `运行数据/pairing-token.txt`，私下填入目标服务的口令，再从仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File apps/mobile/build-android.ps1 -Sdk D:/你的AndroidSDK目录 -Java D:/你的Java21目录
```

默认连接现有公网服务；`-ServiceUrl`可指定另一个地址。`-NoDefaultConnection`生成不预填地址和口令的包，安装后在APP中填写。构建产物在 `apps/mobile/android/app/build/outputs/apk/debug/app-debug.apk`，脚本同时更新仓库上级 `运行数据/distribution/` 分发副本。脚本为本维护电脑预设了部分D盘工具路径；其他电脑需让Node/npm在PATH中可用，并传入自己的SDK和Java目录。

## 二、独立部署模型服务：Linux CPU

以下命令在Linux仓库根目录执行。当前已运行环境为 Python 3.14.4、CPU PyTorch 2.14.0、TabPFN 9.2.0；新增依赖清单按该环境核对。尚未在另一台空白电脑执行整套安装。

### 1. 领取私有模型包

向项目负责人领取整个 `models/app/binary-mean/`，保持内部目录结构。它含主清单、五个完整模型及共享编码器，共43个文件；不能只复制融合权重。TabPFN参考包含训练上下文，模型包只放服务端，不进Git、APK或公开下载目录。

```text
OPMD-Pred/
├─ apps/server/binary_linux_service.py
├─ models/app/binary-mean/manifest.json
├─ models/app/binary-mean/expert0/ … expert4/
└─ runtime/                 # 首次启动时保存记录、附件及口令
```

### 2. 安装依赖

```bash
mkdir -p runtime/cache/pip
export PIP_CACHE_DIR="$PWD/runtime/cache/pip"
python3.14 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0+cpu torchvision==0.29.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r apps/server/requirements-binary-cpu.txt
.venv/bin/python -m pip check
```

清单用于当前二分类推理。原 `requirements.txt`、`requirements.lock` 保留给旧三分类服务；研究训练环境另行配置。

### 3. 准备口令并启动

新建独立服务时运行下面命令；已存在口令时不会覆盖。口令保存在文件中，私下提供给需要连接的组员。

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import secrets
p = Path('runtime/pairing-token.txt')
if not p.exists():
    p.write_text(secrets.token_urlsafe(24), encoding='utf-8')
    p.chmod(0o600)
PY
.venv/bin/python apps/server/binary_linux_service.py
```

服务默认读取 `models/app/binary-mean/manifest.json`；如果存在 `runtime/binary-model-manifest.txt`，以其中路径为准。加载成功后监听 `127.0.0.1:8765`。在另一个终端进入同一仓库，检查：

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

健康接口应返回 `model_id` 为 `sysu-resnet18-tabpfn-transformer-mean5-seed42-20261105`。记录和上传附件默认保存在项目 `runtime/`，不会上传到Git。

### 4. 让前端连接这台服务

该入口只监听本机。跨电脑开发可在前端电脑建立SSH转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 用户名@你的Linux地址
```

然后把前端地址改为 `http://127.0.0.1:8765`，并使用这台新服务的口令。手机跨网络连接需要已有NPC/NPS转发；当前团队公网地址仍为 `www.ming56.cn:9020`。9018是现有个人Linux的SSH端口，9024是NPC连接端口，不是模型HTTP端口。

后台常驻可使用仓库的 `apps/server/opmd-pred.service`。先把其中两处目录改成实际部署目录，并在 `runtime/model-mode.txt` 中写入 `binary`，再安装为systemd用户服务。现有Linux已完成这一步，组员仅开发前端时无需操作。

## 三、协作时改哪里

- 页面与调用：`apps/mobile/src/App.vue`、`apps/mobile/src/api.ts`。
- API与记录：`apps/server/server.py`、`apps/server/core.py`。
- 二分类输入和推理：`apps/server/binary_inputs.py`、`binary_inference.py`；固定模型实现位于 `binary_runtime/`。
- 模型训练：`scripts/research/`、`src/opmd_pred/`；部署不会自动跟随研究代码变化。

接口调用先带 `Authorization: Bearer <口令>`；写入还需请求标识，分析结果需声明支持二分类。这些处理已封装在 `api.ts`，前端优先复用它。

Windows的 `apps/server/launch.ps1` 仍是旧三分类开发入口。要使用当前APP主模型，请连接现有Linux服务或按第二部分部署。
