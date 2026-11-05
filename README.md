# OPMD-Pred

口腔黏膜病损辅助分析项目。更新：2026年11月5日。开发主线为master。

当前线上APP使用五个完整的ResNet-18＋TabPFN＋Transformer二分类模型，各占20%汇总概率，训练Dropout为0.1；输出“正常/轻度”或“中度及以上”。结果页用这套方法六折留出预测中，三个真实病理等级的分数中位数作为参考位置。历史三分类结果继续保留。

组员从[最小部署指南](DEPLOYMENT.md)开始：前端可以直接连接现有Linux服务；独立部署后端需要另行领取私有模型包和口令。源码不内置真实口令，克隆后不会自动连接。

| 目录 | 内容 |
| --- | --- |
| apps/mobile | Vue手机界面、Android工程、APK构建和模拟器入口 |
| apps/server | 后端API、记录保存、模型推理和服务启动 |
| src/opmd_pred | 模型结构、数据处理和训练组件 |
| scripts/research | 当前五模型及图片编码器的训练与复现脚本 |
| scripts/maintenance | 当前模型打包、校验和部署切换脚本 |
| configs | 可共享研究配置；个人配置不入Git |
| models/app/binary-mean | 当前线上五模型二分类的本地私有部署包；只供服务端使用 |
| models/app | 上一层pt文件是旧三分类，供历史复现及本机开发服务使用 |
| data | 项目已有原始数据及processed整理数据；不随本次整理删除 |
| outputs/experiments | 实验模型、原始预测、指标、固定配置及失败结果 |
| outputs/checks | 工程验证与文件整理的原始依据 |
| docs | 方法说明、研究记录和工作记录；成品在仓库外的交付成果 |

本机用户可读成品见仓库上级的交付成果；组员克隆时不会带上这个目录。当前工作区与运行位置见[WORKSPACE.md](WORKSPACE.md)。

源码与配置由Git管理。data、models、outputs、docs和本地配置已忽略，Git克隆不会带来模型、患者资料或连接口令。忽略不等于备份，这些资料的保护方式需要单独维护。
