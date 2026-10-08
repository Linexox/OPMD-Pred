# SYSU实际探索性实验记录

## 当前协议与边界

正式患者身份与两组同图标签核对尚未完成，但原始图片来源已逐SHA定位。为避免停止所有研究，建立独立`sysu-source-name-sha-proxy-v1`探索性协议，保留139条原标签及87张已选图，按原表姓名候选组和共享图片组件一起分配。131为姓名来源组数，不是已核验患者数；两组原始标签冲突明确保留记录。

固定seed42：train97条/91来源组，val20条/20来源组，test22条/20来源组；来源组与图片SHA无跨集合交集。5条完全无输入位于train，训练有效92条；val/test无完全缺失记录。未查看或评估test模型成绩。各方法相同sample-ID分配，字段扩展仅更新CSV和可用性来源。

年龄性别从模型/掩码/token/解释删除；扩展CSV进一步去掉两个原始列。所有图片均来自已选无编号病损.jpg，编号候选图未加入。标签未自动修改，争议记录未删。结果只属来源分组内部探索，不能写成经过身份核验的患者独立临床效果。

## 已自然结束的训练

| 方法 | 分支与训练code HEAD | best/停止epoch | val正确 | 重度召回 | 跨两级错误 | 状态 |
|---|---|---:|---:|---:|---:|---|
| 冻结SAM-DINO +五模态Transformer+Softmax | codex/sysu-source-proxy，49a603b | 5/10 | 12/20=60% | 5/9=55.56% | 4/20=20% | 未test |
| 同配置改CORN | codex/sysu-proxy-corn，ab95f71 | 4/9 | 12/20=60% | 5/9=55.56% | 3/20=15% | 未test |
| 扩展十基因+DNA数量/log编码，Softmax | codex/sysu-proxy-biomarkers，3a081b0 | 5/10 | 11/20=55% | 3/9=33.33% | 3/20=15% | 未test |

基础配置：SAM-DINO B14/518/CLS+patch mean1536冻结；dim128，AdamW1e-4/wd1e-4，batch8、50最大轮/patience5，modality-dropout .15、seed42。先全部概率基线，不叠加成本/一致性/距离正则。最佳checkpoint按全分配val准确率、平局F1选择。

Softmax最佳val CM为`[[7,0,0],[4,0,0],[4,0,5]]`，MAE .60、macro-F1 .4502。CORN为`[[7,0,0],[3,0,1],[3,1,5]]`，MAE .55。扩展字段为`[[7,0,0],[3,1,0],[3,3,3]]`，没有提升本轮验证指标，不能宣称更多基因输入自动有益。

root实际重载完整checkpoint核验history选择与结果，三个模型全部174个encoder tensor精确等于已交付SAM编码器，人口学参数不存在。源CSV、分组文件及manifestSHA记录在各checkpoint。不是只靠静态AST宣称训练完成。

完整模型SHA：

- Softmax：`dd78613062db37cef138ed728b85ecf54ca3f280fdad8b4d8b94f555ee7b892b`。
- CORN：`0879122120d2e71a88ad98127c46e9f7f9df442485acad2ae1de4f26d55e285f`。
- 扩展：`e45bcb950add6af0d05df8c230571d01b49e7f61a3584ca1911be6ef30dca3a8`。

## 固定代价决策

将文档在训练前已定义的研究示例`C=[[0,1,4],[2,0,1],[8,3,0]]`应用于同一个Softmax best5验证概率，没有拟合C或改模型。准确率11/20=55%，重度召回仍5/9，重度→轻度由4变2，跨两级由4/20变2/20，MAE .55。过估由0变3/20，中度召回由0变2/4。该对照说明成本偏好能改变错误方向与距离，同时可能损失准确率；不是医学批准矩阵、特征提升或临床安全保证。

## 实际预测与解释

- 已在20条保存val记录运行best5模型生成[私有离线报告](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/softmax-val-report/report.html)，不嵌原图；显式包含原表参考标签。恢复预测仍12/20正确，与history5一致。
- 真实observed-mask coalition game的SHAP效率最大绝对残差`5.96e-8`，severe目标global InterSHAP约.44665；这不代表44.7%的医学因果作用或临床贡献。
- 固定选择第一个有图像的val ID `level1_007`，不按标签/预测挑例。mean32积分残差-.1625，占输出变化约65.6%，数值未收敛，不推荐定量解读。
- 同病例blur参考32步残差+.06915（约21.2%），增加到256步后残差-.0007716（约0.24%）。[真实IG热图](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/ig_level1_007_blur256/image_ig.png)已视觉查看：原图、signed overlay和带padding边界画布均完整，显示色轴明确；没有声称病灶定位正确。
- blur参考是同一照片Gaussian kernel31/sigma8，仅解释相对于模糊版本的响应，保持所有表格及观测mask固定，不是“图像缺失”解释。数值完整性较好仍不证明忠实性、医学因果或临床校准；需要其他扰动与临床复核。
- 远端初次图像积分已完成并保存tensor，但PNG因matplotlib缺失失败。个人venv通过阿里源安装matplotlib3.11.2后仅续生成图/端点元数据，未重复mean32积分。后续GPU解释均启动前验证空闲。MPLCONFIGDIR、UV_CACHE_DIR后续限定在个人目录。

## 下一步与完成标准

当前无75%+证据，不能标记整体目标完成。模型对中度识别仍弱；优先比较训练类平衡、小参数融合/传统小样本对照和观测子集监督，保持同一开发协议。所有方法先验证，冻结最终方案后才固定test，不用test反复找阈值。

姓名/复诊确认及两组标签处理异步问题仍待用户答复；有新材料后更新正式协议并重跑。已有来源代理成绩不能直接改名成正式结果。
