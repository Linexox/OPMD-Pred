# 按SHA回查原始SYSU已选图片

2026-10-08，按用户补充规则读取`data/sysu`：**文件名为病损.jpg（扩展名不区分大小写）的是已选训练图**；带编号及其他候选图不加入本轮训练。没有重写原始Excel、原始图片或现有processed CSV。

## 实际回查结果

- 原始目录有90份已选`病损.jpg`；当前processed使用87张，87/87均有原始已选文件的SHA256匹配，不是误选带编号候选图。
- 重复组1：`level1_023`与`level2_035`，SHA `c5238edd888eb188f5da6ece8e141d8eb764f960a861f6f0f078b91b018081b6`。
- 重复组2：`level1_049`与`level2_028`，SHA `fa71d29e2f0417065cee5ff5b3f0ed0d722fea6be3f8e82f50f1f82c46bc5617`。
- 两组各自都在原始level1和level2目录存在无编号的已选图，两边文件字节相同。对应Excel的姓名规范化后同名，且各自同level的来源目录名有对应姓名前缀；目录的额外注释使完整目录名不等于姓名，不将这种字符串差异当成误配证明。
- 两组各自的非人口学表格字段并非完全相同。因此可以是重复分类/误挂，也可能涉及不同就诊或检测记录；不能仅凭图片相同就自动选择一个标签、覆盖另一条记录或将两份表格合并。

## 简化身份核对工作

根据原表姓名生成139条的候选分组，共131个姓名组；8组同名共16条需重点核对。编号为`candidate_Pxxx`，没有声称131个已经确认的患者。文件故意使用`candidate_patient_id`而非可直接进入训练的`patient_id`，避免未经确认的映射被误用。

- [原始图片SHA全量回查表](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/processed-to-raw-image-sha-2026-10-08.csv)：原始绝对路径、目录及图片角色，含私有医疗来源信息，仅项目内查看。
- [姓名候选分组](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/name-candidate-groups-2026-10-08.csv)：不包含姓名，用于生成最终匿名映射前审核。
- [8组同名重点核对清单](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/same-name-review-2026-10-08.csv)：样本ID、工作表和原始Excel行号，不需手工重新制作139条表。

“匿名患者映射”不要求医院新建数据库；可基于现有姓名/原始目录生成，经用户确认重复姓名和复诊关系后用于患者分组。原始身份信息只用于追溯分组，不作为模型输入。

## 尚未纳入processed的3份已选图

另有3份原始已选图的SHA不在当前processed中；其中2份没有同level表格姓名前缀候选，1份对应一个已经有另一张已选图的记录`level3_029`。没有直接把额外图片当新患者扩增，也没有替换该记录已有图片。定位见[未收录已选图清单](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/selected-images-not-in-processed-2026-10-08.csv)。

## 保存状态

可复现回查脚本`docs/sysu/trace_raw_images.py`已在本地执行，输出摘要`tmp/sysu-private/raw-image-sha-summary-2026-10-08.json`；这是数据来源审计，不是模型测试。没有启动SYSU训练，没有生成新性能成绩。已向用户请求两组同图异标签是否为不同就诊时点或需纠正记录，暂保留原始标签待核对。
