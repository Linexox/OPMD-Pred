# SYSU foundation 独立静态审阅（2026-10-08）

审阅目录：`C:/Users/aljcn/.codex/worktrees/sysu-foundation/OPMD-Pred`，分支 `codex/sysu-foundation`。本报告只依据工作区源码阅读；未运行import/module、语法编译、tests、模型恢复或训练。实施中的SHAP/InterSHAP、analyze和可视化文件不在本次审阅范围。源码仍在协调者修改中，以下不是提交快照或功能验收。

## 结论

新入口在结构上解决了旧版主要数据协议问题：不用记录ID推定患者、同图标签冲突拒绝拆分、人口学不进入模型、训练词表和完整manifest/checkpoint、自然无输入病例分母显式保留。静态阅读未发现新训练/评价核心路径的确定P0或P1运行错误。

**这不证明实际模型可训练、编码器已成功恢复、患者身份真实或SYSU达到75%。** 原数据同图异标签和可信患者/就诊映射仍是实际评价的前置问题。代码不能凭manifest里的字符串自行证明身份已经人工核实。

## 逐项检查

| 要求 | 源码证据与静态结论 | 未验证的边界 |
|---|---|---|
| 禁用年龄/性别 | `data/sysu.py`MODALITIES只有image/history/tct/dna/methylation；dataset无demo键；fusion无demo encoder/token；dropout按选定五模态 | 历史CSV仍可读age/sex追溯字段，不进入dataset输出；未通过真实运行追踪所有输入 |
| 可选模态 | 新CLI限制五种名字且拒绝重复；dataset将未选字段置零/false；fusion只生成所选token；不选image时不构建图像模型 | 仍为未选的数字/TCT模态分配一些未使用参数，效率可精简，但未见输入泄漏 |
| 真实患者映射 | `build_manifest`强制每个sample显式映射patient_id，未知/重复/空ID拒绝；不自动推定；同图同标签的patient组件只约束共同split | 无法确认外部patient_id真的来自医院；需要人工映射，不能填sample_id冒充身份 |
| 图片/标签冲突 | `inspect_records`每张图计算SHA，异标签同SHA立即拒绝；`validate_manifest`重新查CSV SHA、标签、图SHA、patient/hash跨split及记录完整性 | 当前只字节重复，未检测压缩/裁剪近重复；原始冲突未由代码纠正 |
| 无输入分母 | 每行包括全缺失行先分配split；`select_records`只排除对应模态无观测者；coverage保留assigned/eligible支持；`accuracy_all_assigned`将abstention计未正确 | no-input没有产生伪个体预测是合理的；病例覆盖率和条件性能必须一起展示 |
| Train-only TCT | 新训练只在eligible train建词表，missing=0、未知观测=1、训练类别从2开始；eval使用checkpoint词表 | 词表不会按val/test重fit；未验证真实loader/embedding调用 |
| split冻结与恢复 | checkpoint含完整manifest及canonical digest、源文件digest、有效ID、exclusion、coverage；eval重查真实CSV/图并比对有效ID；使用保存model_config严格载state | manifest初始人工审核与来源文件归档仍靠流程；版本标签不替代真实代码提交SHA |
| DINO输入/描述符 | `ImageEncoder(kind=dinov2_b14)`创建同名timm B14模型且不下载；forward_features后CLS+全部patch均值concat=1536；transform用PIL.pad518、BILINEAR、居中RGB124/116/104及ImageNet norm；load_artifact验证model/size/dim/pad/norm，strict恢复encoder | 当前纯SAM artifact已由上一阶段root审计；本次没有加载artifact或验证模型实际forward |
| 冻结/微调 | freeze会关闭梯度且保持encoder eval；unfreeze合法block范围后解冻末尾block及末端norm；新训练default0 | SYSU解冻末端norm是后续训练选择，不代表Zenodo导出时norm曾被训练；GPU资源检查仍由启动流程负责 |
| Head/代价方向 | 有序threshold gap softplus保持顺序；三类概率构造合理。`probabilities @ C`再argmin符合C行真实、列行动；代价矩阵检查3×3、非负有限、对角0 | 未提供医学代价，不应把默认argmax或任意代价说成已保证重度召回；代价不改变当前训练loss/epoch选择 |
| 全缺失dropout | subset空集合安全跳过；p为触发概率；Bernoulli全drop时随机保留一项，消除了固定首项偏置 | 数据正常no-input已被排除；p不应被解读为subset模式中的平均缺失比例 |
| 旧接口 | ImageEncoder仍默认ViT-B16，旧入口保留参数且给exploratory warning；新checkpoint格式明确拒绝旧人口学模型 | 不承诺旧6模态checkpoint与新5模态state兼容；旧接口仅探索，不可替代患者协议 |

## 已报告且正在处理：评价单位

`train_sysu.run_epoch`的loss、概率与混淆矩阵按CSV记录计算；patient_id只用于split约束，没有patient/encounter聚合。若同一患者多图或多次就诊，更多记录患者的权重更大。因此**患者独立split下的记录级accuracy，不等于患者级accuracy**。若同一患者跨次就诊有不同等级，直接平均其概率或统一标签会改变临床任务，不能自动实施。

该问题已通知root。root决定本轮明确输出`evaluation_unit=record`，coverage补充eligible和fully-abstained患者人数，不未经医学确认实现跨就诊聚合。审阅最后读取时`sysu_protocol.split_coverage`已加入`eligible_patient_count`及`fully_abstained_patient_count`；评价单位标注修改由root完成后应再核对源码。基础实现文档已经将accuracy标成eligible记录条件指标。

若比赛或论文要求患者级终点，还需确认patient/encounter映射、每次就诊病理时间与标签策略，再设计患者级预测聚合/cluster bootstrap。当前输出不能作为该终点的完成证据。

## 剩余P2建议

1. **代价决策同分母严重召回。** `add_coverage_metrics`为argmax提供`severe_recall_all_assigned`；cost_decision只补了`accuracy_all_assigned`，其severe_recall仍是eligible条件召回。建议同步补cost的全assigned重度召回，并标明两套指标决策来源，避免missing模式改变分母而让代价结果看似更好。
2. **Legacy fusion词表语义。** 旧`train_multimodal --fusion-checkpoint`严格载fusion state，却未比对/恢复checkpoint的TCT vocab。不同训练seed可能使同大小词表对应不同字符串，shape检查不能发现语义错位；可直接拒绝不一致词表。该问题不影响没有此选项的新train_sysu。
3. **Artifact元数据可更完整。** 当前loader核对模型、size/dim、padding和norm，但未核对descriptor/preprocessing字符串（保比全图、插值、CLS+patch mean）及public revision/source weight SHA。新训练已保存实际artifact SHA；建议同时归档这些已有元数据，在不同来源artifact间选择时减少混淆。当前SAM artifact正确性沿用上一阶段真实核验，不能由此静态报告重新“证明”。
4. **评价类支持与置信区间。** 组件随机split明确没有类别分层保证，新训练只要求train三类齐全；val/test可能缺某类，此时重度指标null。临床召回目标需要足够且真实的重度病例支持；不能挑测试成绩选seed。当前尚无cluster置信区间/开发集分组CV实施，后续评价计划应补齐。
5. **数据准备的review gate。** 新prepare显式工作表、新目录且遇多候选目录停止，解决旧active与覆盖问题。原数据已确认存在候选歧义，当前脚本可能正确地拒绝准备，而不是一键重新产出全部139条。需要经审核的图像绑定表。脚本没有完整source workbook/copy manifest输出，可进一步保存来源SHA与明确sheet/row关联。
6. **CLI参数与可复现元数据。** 当前保存软件版本和配置，但未保存当前Git SHA；weight-decay未显式非负/finite检查，lr只检查正数而未检查finite。属于实验入口完善，不作为实际训练已失败的证据。

## 审阅后的完成边界

已完成：独立静态审阅和上述发现反馈，原数据/源码未由本审阅者修改，未接触远端GPU。尚未完成：真实患者身份核验、冲突病理归属确认、当前source artifact运行恢复、真实训练/冻结测试、75%及召回提升证据。后续不得以本报告或语法编译替代功能验证和医学评价。

## 总协调者跟进（审阅之后）

已补evaluation_unit=record、eligible/fully_abstained患者数；代价决策的全分配重度召回；legacy fusion词表语义校验；encoder来源public revision/SHA与预处理字段。未实现未经医学确认的患者/就诊等级聚合，记录指标不能叫患者级准确率。prepare脚本现在新目录输出、显式sheet及歧义绑定报错，尚未执行。

总代码静态AST解析27个Python文件、git diff --check与uv lock --check成功。这些均未执行模型、数据协议、梯度或训练；独立静态审阅结论不等于运行验证。InterSHAP后续修正由总协调者依据原文公式4–11静态复核，不属于本独立审阅范围。
