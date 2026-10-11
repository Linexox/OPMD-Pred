# 手机APP本地服务

复用 `patient_demo/core.py` 的就诊、任务、来源确认及SQLite规则，复用 `inference.py` 的固定模型。数据目录独立于桌面Demo。

## 启动

当前电脑使用 `launch.ps1`，或双击根目录外的“启动手机APP服务.cmd”。默认端口8765、目录 `%LOCALAPPDATA%/OPMDMobileDemo`。首次生成私有配对口令，后续启动继续使用。

新电脑需安装主项目/桌面研究推理依赖，以及本目录 `requirements.lock` 中服务依赖；配置同级 `patient_demo/models.local.json` 的两份冻结模型路径。

```powershell
python server.py --host 0.0.0.0 --port 8765 --data-dir C:/private/opmd-mobile --model-config C:/private/models.json
```

## 接口

所有请求使用 `Authorization: Bearer <配对口令>`；写请求还需 `X-Request-Id`。相同请求标识与内容返回已归档响应，内容改变返回409。编辑、提交、复核和推理需当前 `revision`。

| 能力 | 路由 |
|---|---|
| 连接、字段选项 | GET /api/health，/api/options |
| 档案 | GET/POST /api/patients |
| 就诊、任务列表 | GET /api/patients/{id}/visits，/tasks |
| 附件 | POST /api/assets；GET /api/assets/{id} |
| 记录 | POST /api/visits；GET/PATCH /api/visits/{id} |
| 提交、医生反馈 | POST /api/visits/{id}/submissions，/reviews |
| 关联补充、复诊自报 | POST /api/tasks/{id}/submissions，/completions |
| 研究推理 | POST /api/visits/{id}/runs；GET /api/runs/{id} |
| 历史运行 | GET /api/visits/{id}/runs |

附件上传是JSON `{name,kind,data}`，data为Base64原始字节，kind为photo或report。保存记录使用photo_id及新增report_ids，保留既有附件；不接受手机指定服务器文件路径。

字段名沿用现有snake_case。错误使用 `{error:{code,message,field}}`；输入错误422、口令错误401、版本冲突409。任务/就诊保持原UUID与状态枚举；详情中的photo/reports只返回附件ID和名称。

SQLite连接在发起操作的线程建立，操作串行保护；推理线程独立且单任务运行，使用启动时的就诊快照。研究任务queued/running/predicted/abstain/failed；原输入版本与SHA保留。服务重启将中断任务标为失败，允许重试。

归档SHA匹配是加载条件；没有模型、全缺失、未知TCT等继续遵循原接口。模型仍是seed42基线fold0，未替换为距离损失候选。
