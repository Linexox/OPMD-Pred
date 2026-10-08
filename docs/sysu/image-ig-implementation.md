# SYSU 图像 Integrated Gradients 实现

状态：代码已写，未运行模块、模型前向、病例解释或训练。本轮仅 AST 静态解析与 diff 检查；尚无数值有效性或临床解释证据。当前缺患者核验、图像标签冲突等前置问题仍阻止可信训练。

## 方法及接口

依据 [Sundararajan 等，Axiomatic Attribution for Deep Networks，ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)。不引入 Captum；`analysis/image_attribution.py` 的 `integrated_gradients(model,batch,target,baseline,steps)` 对归一化图像的直线路径采用梯形积分，steps 为区间数，计算 steps+1 次图像梯度。

- 单张真实已观测图像；模型必须 eval，完整图像编码器路径参与输入梯度，冻结权重不妨碍输入求导。只调用 `autograd.grad(..., image)`，不 backward、不更新参数、不累计参数 `.grad`、不使用 detach 图像缓存绕过编码器。
- 其余病史/TCT/DNA/甲基化数值和观测 mask 全程固定；`image_present=True`。均值/黑色参考仅为图像内容参照，**不是缺失图像**。
- 目标 severe=`P(重度)`，at_least_moderate=`P(中度)+P(重度)`，expected_severity=`P(中度)+2P(重度)`；由相同 `model_probabilities` 支持 softmax/ordinal checkpoint。
- mean 基线为 normalized zero，对应 RGB ImageNet 均值；black 基线为 `-mean/std`，对应真实 RGB 黑色。不同基线可能得到不同归因，必须在解释里显示。
- 返回未归一化 `[3,H,W]` signed 像素贡献，单位是目标概率或级别分值。完整性残差为 `f(image)-f(baseline)-sum(IG)`；它受积分步数和数值误差影响，小残差不证明解释忠实或临床正确。

## CLI：未来获准数值核验后才运行

```bash
PYTHONPATH=src python -m opmd_pred.scripts.explain_sysu_image \
  --csv /path/sysu.csv --images /path/images \
  --checkpoint /path/sysu_checkpoint.pt \
  --sample-id VALIDATION_SAMPLE_ID --output /path/new_ig_output \
  --target severe --baseline mean --steps 32 --device cpu
```

仅接受 checkpoint 保存的 **val sample ID**，没有 train/test 自动选择；无输入或无实际图像报资料不足。复用 `read_checkpoint`、`validate_manifest`、`select_records`、`assign_tct_ids` 和 `restore_checkpoint`。先 CPU 验证 CSV/image SHA 及有效 split/exclusions，后恢复模型到设备；无外部预训练下载，默认 CPU。输出目录必须不存在，禁止覆盖。本轮没有生成任何医院病例图片或解释输出。

文件：`pixel_attribution.pt` 为模型输入坐标的原始 tensor；`image_ig.png` 为静态 signed 热图；`metadata.json` 记录 checkpoint/image SHA、manifest digest、目标、参考、步数、端点目标值、残差、原图尺寸、模型画布与内容框、padding signed/absolute贡献、显示比例与 Pillow 版本。

## 原图映射与 padding

[Pillow 官方 ImageOps.pad 源码](https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageOps.html#pad) 先 contain，再以 `round((目标尺寸-缩放尺寸)*0.5)` 居中。DINO B14 用与数据变换相同的 contain(BILINEAR)、518画布和124/116/104填充几何；维持其 CLS+patch mean 路径，包括 padding 的影响。ViT B16 是224×224拉伸，单独记录该变换。

热图按 RGB 通道求 signed sum，以整个模型画布的绝对最大值为对称色轴，没有只截取正值或每个区域独立缩放。先裁出内容框，再用 PIL bilinear 映回原图尺寸，仅用于显示，**不是保贡献和的数学变换**；数值完整性始终按原始模型像素 tensor 计算。模型画布上明确画内容边界，padding贡献单独保存，不把填充区域映射成口腔组织。

## 待验证事项

未来需要步数收敛、有限差分/遮挡扰动、参数随机化及基线敏感性检查，以及医学导师复核。图像参考本身可能分布外；IG不等同病灶定位，未获得病灶标注不能给定位IoU，更不能宣称医学因果或稳定临床效果。当前只有实现和静态语法证据。
