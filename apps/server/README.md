# APP后端

更新：2026年11月5日。

本目录提供档案、就诊、附件、补资料、医生反馈和模型分析接口。手机界面通过api.ts调用，服务保留SQLite和附件。

| 使用位置 | 入口和模型 |
| --- | --- |
| Linux线上服务 | launch-linux.sh；model-mode.txt为binary时运行binary_linux_service.py，活动清单由runtime/binary-model-manifest.txt选择，当前为models/app/binary-mean/manifest.json |
| Windows本机开发服务 | launch.ps1，或医工目录的启动手机APP服务.cmd；仍使用models.local.json配置的旧三分类 |

Linux主模型是五个完整ResNet-18＋TabPFN＋Transformer二分类模型，各占20%汇总预测，训练Dropout为0.1。固定推理实现位于binary_runtime，当前参数包为models/app/binary-mean，每个完整模型使用自己的5份TabPFN参考，共25份；不是只平均表格分支。单模型旧包models/app/binary仍保留原位置用于恢复。正式推理不依赖checks目录。

Windows数据默认在D:/.Workspace/医工/运行数据。Linux数据仍在部署根目录runtime。两者都包含records.sqlite3、附件、私有连接口令和distribution分发副本；不能把整个运行目录交给前端或评委。

主要接口：/api/health检查服务；/api/patients管理档案；/api/visits保存就诊；/api/assets上传附件；/api/visits/{id}/runs发起分析；/api/runs/{id}查询状态；/api/visits/{id}/runs读取历史。写入使用请求标识与资料版本；连接口令放Authorization头，源码中不包含实际口令。

二分类结果返回两个分数和类别名称，另外返回grade_reference供前端显示三个中位数参考点。参考点不改变二分类结果。旧三分类历史记录按其原类别读取；旧客户端不支持二分类格式时会收到更新提示。

组员连接与独立部署步骤见根目录[DEPLOYMENT.md](../../DEPLOYMENT.md)。当前二分类依赖为requirements-binary-cpu.txt，按实际Linux环境核对；requirements.txt和requirements.lock保留给旧三分类入口。实际维护位置见WORKSPACE.md。
