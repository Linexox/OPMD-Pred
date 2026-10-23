# 手机APP本地服务

手机接口、就诊记录、任务管理、模型推理和解释集中在本目录。原 `mobile_service` 与 `patient_demo` 已合并，旧桌面界面不再维护。

## 启动

当前电脑使用 `launch.ps1`，或双击根目录外的“启动手机APP服务.cmd”。默认端口8765、目录 `D:/.Workspace/医工/OPMDMobileDemo`。首次生成私有配对口令，后续启动继续使用。

新电脑需安装主项目及本目录 `requirements.txt` 中的服务和推理依赖；`requirements.lock` 记录现有API依赖版本，推理依赖列于 `requirements-inference.txt`。本机模型配置为本目录的 `models.local.json`，默认引用根目录 `models/app` 下的两份模型，不需要PyQt。

```powershell
python server.py --host 0.0.0.0 --port 8765 --data-dir D:/.Workspace/医工/OPMDMobileDemo --model-config D:/.Workspace/医工/OPMD-Pred/apps/server/models.local.json
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

2026-10-22：默认数据目录改为仓库同级的 `D:/.Workspace/医工/OPMDMobileDemo`。`--data-dir` / `-DataDir` 可显式指定其他目录；未指定时不再使用C盘用户目录。现有记录、附件、口令和APK已迁移。

2026-10-22界面整理：服务不再自动生成预填就诊资料。日常档案列表排除历史生成档案，数据库不删除原记录。为旧APK保留的示例查询返回空列表，旧载入入口返回410并提示手动新建档案。

2026-10-22上传修复：接收文件前确保uploads目录存在；无法写入时返回明确的503存储错误。现有照片和就诊输入不变。

2026-10-22检测字段调整：TCT、DNA和甲基化按已保存的填写值参与分析，不再要求额外确认报告来源。旧APK传入的report_confirmed字段忽略，旧记录中的同名状态不影响推理。报告附件与来源说明选填；数值合法性检查保留。DNA仍需异常细胞数、总细胞数和DI阈值，真实零值不视为缺失。新运行记录写入input_policy=entered_fields_v1，旧分析结果保持原样。
