# 保留原标签的来源分组探索性基线

该分支`codex/sysu-source-proxy`基于`codex/sysu-research`，目的是在最终身份/标签核对尚未完成时获得探索性训练证据。它不替代正式研究终点，不自动宣称经过核验的患者独立或临床结果。

## 数据边界

使用原有139条CSV及87张处理后图片；87/87图片SHA已回查到原始目录的无编号病损.jpg，编号候选图不加入。保持原标签，包括两组同图跨正常/轻度与中度记录；没有挑一个标签覆盖另一条，也没有删除争议记录。5条全无观测记录保留在分配分母，模型不为它们生成个体预测。

姓名候选组来自原始三份Excel的指定工作表，规范化空白及已有new_/v/删除_前缀规则；共有131个姓名组，8组同名16条。名称不是经医院核验的身份。相同姓名和任何相同图片SHA组件必须整体划到同一集合，已知重复不跨train/val/test；仍可能存在异名同人或近重复，不能承诺完全患者独立。

## 独立协议而非放宽正式协议

协议ID为`sysu-source-name-sha-proxy-v1`，只接受具有`sample_id,candidate_patient_id,review_status`的未审核来源候选文件。manifest明确`identity_verified=false`、`labels_corrected=false`、原始CSV及分组文件SHA、两组未解决图片标签冲突、所有记录/图片SHA、来源组划分。

正式`sysu-patient-hash-v1`入口在其他分支保持审核要求。这两种manifest不能互相当成同一研究协议，proxy checkpoint需本分支源码读取。输出字段使用source_group_count，不称131个已核验患者。

## 第一组固定方法

固定分组seed42、训练seed42，val/test各15%来源组件；只train/val，不自动test。SAM-DINO B14/518/1536冻结，五模态dimension128、Transformer融合、softmax CE、AdamW1e-4/wd1e-4、batch8、50epochs/patience5、Bernoulli模态dropout .15。所有新研究正则默认0，不设成本矩阵。选checkpoint仍为全分配validation argmax accuracy，平局有效记录macro-F1。

这是开发阶段的基线，不把未核对标签或来源身份假设隐藏在成绩中。记录准确率、重度召回/precision、低估/跨两级错误、coverage及全部分配分母；以后核对身份或标签若改变，必须新版本manifest及新实验，旧成绩不混入正式终点评价。

每次GPU启动前即时检查无其他compute任务，远端代码/输入/输出均在个人24334067目录。训练命令、实际PID、console和code HEAD在输出旁保存；既有环境与最佳encoder复用，不下载新权重，不运行额外unit tests。
