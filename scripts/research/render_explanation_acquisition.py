"""Render concise public results and one research figure from audited private evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/"outputs/experiments/sysu/explanation-acquisition-20261017/attempt1"
PUBLIC=ROOT/"docs/daily/2026-10-17"
LABELS=["正常/轻度","中度","重度"]
NAMES=["照片","病史","TCT","DNA","甲基化"]
POLICIES={"fixed":"固定顺序","random":"随机选择","historical_shap":"按历史SHAP贡献","historical_interaction":"按历史成对交互"}


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(value):return "—" if value is None else f"{value*100:.2f}%"


def table(data):
    lines=["| 补一次的选择方式 | 准确率 | 中度正确识别 | 重度正确识别 | 重度判成正常/轻度 ↓ |",
           "|---|---:|---:|---:|---:|"]
    for policy,label in POLICIES.items():
        r=data[policy]
        lines.append(f"| {label} | {pct(r['accuracy'])} | {pct(r['moderate_recall'])} | {pct(r['severe_recall'])} | {pct(r['severe_to_low'])} |")
    return "\n".join(lines)


def main():
    result=read(PRIVATE/"results.json");audit=read(PRIVATE/"independent-audit.json")
    if not audit["passed"] or audit["results_sha256"]!=sha(PRIVATE/"results.json"):raise ValueError("Results have not passed audit")
    preparation=read(PRIVATE/"preparation-summary.json");inference=read(PRIVATE/"inference-summary.json")
    main_result=result["tables"]["ordinal_pretrain"]["primary"]["mean"]
    ref_result=result["tables"]["balanced_reference"]["primary"]["mean"]
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,
        "font.size":12,"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
    figure,(left,right)=plt.subplots(1,2,figsize=(13.8,6.6),gridspec_kw={"width_ratios":[1.08,1]})
    colors=["#7B61B5","#4294A8"];y=np.arange(5)
    for offset,(method,label) in enumerate([("ordinal_pretrain","有序分类主模型"),("balanced_reference","三分类参照")]):
        rows=read(PRIVATE/"oof_explanations"/f"{method}.json")
        per_seed=np.array([[np.abs(r["per_seed"][s]["shap"]) for r in rows if any(r["observed"])] for s in range(3)]).mean(1)*100
        means=per_seed.mean(0)
        left.barh(y+(offset-.5)*.32,means,height=.29,color=colors[offset],label=label,
            xerr=np.stack([means-per_seed.min(0),per_seed.max(0)-means]),error_kw={"ecolor":"#77717e","capsize":3,"linewidth":1})
    left.set_yticks(y,NAMES);left.invert_yaxis();left.set_xlabel("对重度分数的平均影响幅度（百分点）")
    left.set_title("哪些资料影响模型分数？",fontsize=16,pad=20);left.legend(loc="lower right",frameon=False,fontsize=11)
    left.set_axisbelow(True);left.grid(axis="x",alpha=.16);left.set_xlim(left=0)
    matrix=np.array(result["explanations"]["ordinal_pretrain"]["mean_abs_pairwise_interaction"])*100
    masked=matrix.copy();np.fill_diagonal(masked,np.nan)
    cmap=plt.colormaps["Purples"].copy();cmap.set_bad("#f1eff4")
    im=right.imshow(masked,cmap=cmap,vmin=0,vmax=float(np.nanmax(masked))*1.1)
    right.set_xticks(y,NAMES);right.set_yticks(y,NAMES);right.set_title("哪些资料共同影响分数？",fontsize=16,pad=20)
    for i in range(5):
        for j in range(5):
            if i==j:right.text(j,i,"—",ha="center",va="center",color="#99929e")
            else:right.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",color="white" if matrix[i,j]>np.nanmax(masked)*.62 else "#3f354b",fontsize=11)
    bar=figure.colorbar(im,ax=right,fraction=.045,pad=.04);bar.set_label("平均成对交互幅度（百分点）",fontsize=11)
    figure.suptitle("当前模型解释：影响大小与预测是否正确分别评价",fontsize=18,x=.48,y=.98)
    figure.text(.06,.055,"112份有输入的记录，三套保存模型；原缺失项贡献为零。误差线表示三套模型的范围。",fontsize=11,color="#625c6b")
    figure.text(.06,.018,"这里解释重度分数的变化，不代表诊断准确率、检查必要性或医学因果。右图为有序分类主模型。",fontsize=11,color="#625c6b")
    figure.subplots_adjust(left=.07,right=.94,top=.83,bottom=.22,wspace=.32)
    figures=PUBLIC/"figures";figures.mkdir(parents=True,exist_ok=True)
    figure.savefig(figures/"explanation-acquisition.png",dpi=180);figure.savefig(figures/"explanation-acquisition.pdf");plt.close(figure)
    public={"scope":preparation,"tables":result["tables"],"explanations":result["explanations"],"gates":result["gates"],
        "audit":audit,"inference":{k:v for k,v in inference.items() if k!="logs"},"source_results_sha256":sha(PRIVATE/"results.json"),
        "training":False,"app_changed":False,"clinical_user_study_completed":False}
    (PUBLIC/"explanation-acquisition-results.json").write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding="utf-8")
    delta=(main_result["historical_interaction"]["accuracy"]-main_result["fixed"]["accuracy"])*100
    text=f"""# 补资料与模型解释：本轮实际结果

2026年10月17日。**补回资料有帮助；按历史交互选择下一项，没有达到事先约定的继续条件。APP本轮不加入自动推荐。**

## 1. 实际做了什么

使用现有两类模型、各三套训练种子及固定五折，核对原预测，重算SHAP及成对交互，并完成四种补充方式的比较。没有重新训练、修改原标签或评价旧22份测试资料。

主比较包含90份资料记录：正常/轻度42份、中度24份、重度24份。先按固定规则暂时隐藏实际已有资料，所有方法使用同一起点、相同候选，只补一次。**这是假设模型暂时没有收到部分资料的离线实验，不是真实患者试用。**

## 2. 补回资料能纠错，也可能把判断改错

主模型沿用五模态、对比预训练和有序分类，即分别判断是否超过轻度、是否超过中度。以下是三套保存模型分别计算指标后的平均，不是合并分数再判断：

- 暂时隐藏部分资料后，准确率为 **{pct(main_result['before']['accuracy'])}** 。
- 按训练资料确定的固定顺序补一次，准确率为 **{pct(main_result['fixed']['accuracy'])}** 。
- 使用全部已有资料，准确率为 **{pct(main_result['full']['accuracy'])}** 。

这说明补充资料存在改善空间。但一项资料能纠错，也可能把原本正确的判断改错，不能只展示成功例子。以上三项是同一90份资料的比较，不能拿来宣称模型超过历史项目成绩。

## 3. 怎样选，实际比较结果

以下均为同一90份记录、三套模型平均；中度和重度分别以24份该类记录为分母。随机方法计算候选动作等概率选择的平均表现。

{table(main_result)}

**历史交互比固定顺序只提高{delta:.2f}个百分点；重度判轻仅在一套模型下降，另外两套相同。** 与随机选择相比，它的中度识别更低，未通过预先确定的继续条件。更换为全部合法起点、先在每份记录内部平均后，也没有胜过固定顺序。

SHAP选择的准确率略高，但中度识别进一步下降，也不能只凭最高准确率选用。

在APP同类三分类模型上，历史交互选择的准确率为{pct(ref_result['historical_interaction']['accuracy'])}，固定顺序为{pct(ref_result['fixed']['accuracy'])}；交互选择低{(ref_result['fixed']['accuracy']-ref_result['historical_interaction']['accuracy'])*100:.2f}个百分点，中度识别也更低。因此结论没有稳定跨模型成立。

## 4. 解释分析发现什么

![模型资料贡献和成对交互](D:/.Workspace/医工/OPMD-Pred/docs/daily/2026-10-17/figures/explanation-acquisition.png)

- 对模型重度分数而言，TCT和照片的平均影响幅度较大。
- 同一份记录换用三套保存模型时，DNA和甲基化贡献方向经常变化，提示模型对这些输入的利用不够稳定。
- 实际中度或重度却判成正常/轻度的情况中，TCT、照片或病史可能把重度分数向下推。需要继续核对编码、原报告及分级方式；不能因此认定这些检查没有医学价值。

**解释可以帮助找到排查方向，但本轮没有证据证明高交互就等于补充收益高。** 当前有序模型的中度识别问题，仍是比增加推荐模块更需要先解决的问题。

## 5. 四个真实计算例子

例子按预先规定的错误类型和记录顺序选取，不代表医生已经看过或纠正过。

| 例子 | 现有参考等级 | 本轮看到的变化 |
|---|---|---|
| A | 重度 | 全部已有资料仍判正常/轻度；照片和病史下拉重度分数，DNA和甲基化上推，但不足以改变最终分级 |
| B | 中度 | 全部已有资料仍判正常/轻度；TCT和病史下拉重度分数 |
| C | 正常/轻度 | 原先判重度，补回一份TCT＋DNA联合资料后判正常/轻度，纠正了错误 |
| D | 中度 | 原先判中度，补回病史后反而判正常/轻度，产生了错误 |

C、D描述模拟恢复已有数据，不是患者实际提交或医生干预。病例编号、各项分数和原始证据位置留在私有文件，不在公开说明中复制患者资料。

## 6. 本轮结论与后续

1. **保留解释分析作为模型排查工具。** 优先看TCT表述、DNA/甲基化利用不稳定以及中度分级方式，不能把SHAP方向直接解释成医学机制。
2. **停止把这版历史交互选择接入APP。** 两类模型均未通过原定条件；本轮不继续搜索更复杂的推荐网络或换模拟起点追成绩。
3. 临床上哪些资料真的难取得、需要等待，当前仍没有患者操作证据。已持有且容易提交的报告应尽量一次补齐；后续要证明效率，仍需真实操作评价。

本轮完成3篇方法部分阅读，另1篇医学协作研究只能核对摘要；[文献借鉴与阅读范围](D:/.Workspace/医工/OPMD-Pred/docs/daily/2026-10-17/literature-findings.md)单独记录。没有医生使用收益，也没有新的临床验证成绩。

<details>
<summary>三套种子、另一模型、资料范围和计算核对</summary>

### 当前有序模型：三套保存模型分别是多少

| 训练种子 | 固定顺序准确率 | SHAP准确率 | 交互准确率 | 固定顺序重度判轻 | 交互重度判轻 |
|---|---:|---:|---:|---:|---:|
"""
    for seed,row in zip([42,43,44],result["tables"]["ordinal_pretrain"]["primary"]["per_seed"]):
        text+=f"| {seed} | {pct(row['fixed']['accuracy'])} | {pct(row['historical_shap']['accuracy'])} | {pct(row['historical_interaction']['accuracy'])} | {pct(row['fixed']['severe_to_low'])} | {pct(row['historical_interaction']['severe_to_low'])} |\n"
    text+=f"""
### APP同类三分类参照：相同90份资料

{table(ref_result)}

### 不同范围的结果均保留

| 范围 | 记录数 | 正常/轻度、中度、重度 | 主模型固定顺序准确率 | 主模型交互准确率 |
|---|---:|---|---:|---:|
"""
    scope_names={"primary":"预先固定的起点","material_mapping_recorded":"报告来源能对应的子集","without_prelisted_quality_flags":"排除预列疑点的子集","all_legal_starts_record_weighted":"全部合法起点，记录内平均"}
    for key,label in scope_names.items():
        s=result["tables"]["ordinal_pretrain"][key]["mean"]
        text+=f"| {label} | {s['fixed']['records']} | {', '.join(map(str,s['fixed']['class_counts']))} | {pct(s['fixed']['accuracy'])} | {pct(s['historical_interaction']['accuracy'])} |\n"
    inter=result["tables"]["ordinal_pretrain"]["primary"]["intervals"]["fixed"]
    text+=f"""
疑点审查主要覆盖中度，因此排除疑点后中度从24份减为11份。这个子集组成明显改变，不能把其中更高的准确率解释为清洗提高了模型能力。

主模型交互与固定顺序的准确率差为{inter['accuracy']['difference']*100:+.2f}个百分点，按现有来源组重采样得到的95%范围为{inter['accuracy']['lower']*100:+.2f}至{inter['accuracy']['upper']*100:+.2f}个百分点，包含没有改善的情况。这只是开发资料上的探索性不确定性估计。

### 范围、动作与限制

- 117份开发资料中，112份至少有一类输入；90份有足够的补充选择。59份记录的TCT与DNA能追到同一份匹配报告，已合并为一个动作；不是所有117份都有联合报告。
- 90份中70份的模型资料类型可对应到报告来源；这不表示医院已经确认同次就诊、同部位及全部字段。20份来源映射不足的记录，仅用于模型信息补充分析，不用来宣称减少患者上传次数。
- 病理等级、报告区间、多标本和时间疑点预先标记；原标签和文件未改。原模型仍继承原训练标签问题，本轮不能消除它们。
- 主比较保留非空初始资料，无输入拒判为0；完整117份原始评价仍有5份全缺失拒判。没有把全缺失参考分数返回为患者诊断。
- 仅用该折训练病例形成固定顺序、SHAP和交互历史。选择函数不接收本人的隐藏数值或病理标签。30份模型的全部组合共112320组三类分数，其中包含训练记录的历史估计及全缺失数学参考；不是112320份病例。
- 两类模型各自15折、三套种子；没有重复训练三次，也没有在新结果中挑最好模型。当前比较是人为隐藏情境，不能与旧21份验证的71.43%或其他人群直接比较。

### 独立核对与归档

- 672组原留出预测重算，最大概率误差{inference['maximum_probability_error']:.2e}；全部30份模型检查通过。
- 另用NumPy独立重算贡献与交互、训练病例统计、540份记录—模型的选择结果、随机期望和120项纠错/改错守恒关系；核对通过。
- 过程中发现两个汇总细节：全部起点的纠错数应先逐起点计算再在记录内平均；完整117份包含拒判，分级误差应以有预测资料为分母。已修正，初稿和修订说明保留；四种方法的主准确率及严重错误指标未变。
- 本轮只使用CPU，组合前向约{inference['elapsed_seconds']:.2f}秒；这是复用冻结图像特征后的计算时间，不含代码编写、论文阅读和核对。没有GPU训练。
- [完整聚合结果](D:/.Workspace/医工/OPMD-Pred/docs/daily/2026-10-17/explanation-acquisition-results.json)、[执行前计划](D:/.Workspace/医工/OPMD-Pred/docs/daily/2026-10-17/experiment-plan.md)。
- 病例逐份、完整模型指纹、命令、历史估计及缓存位于 `outputs/experiments/sysu/explanation-acquisition-20261017/attempt1/`。图可下载[PDF](D:/.Workspace/医工/OPMD-Pred/docs/daily/2026-10-17/figures/explanation-acquisition.pdf)。

</details>
"""
    (PUBLIC/"explanation-acquisition-results.md").write_text(text,encoding="utf-8")
    print(json.dumps(dict(report=str(PUBLIC/"explanation-acquisition-results.md"),figure=str(figures/"explanation-acquisition.png"),audited=True),ensure_ascii=True))


if __name__=="__main__":main()
