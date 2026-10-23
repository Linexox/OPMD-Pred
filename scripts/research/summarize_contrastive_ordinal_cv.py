"""Create a concise Chinese report from completed, independently audited runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


LABELS = {
    "balanced_reference": "当前类别平衡三分类",
    "softmax": "三分类，不加类别权重",
    "ordinal": "仅改有序分类",
    "softmax_pretrain": "对比预训练＋三分类（不加权）",
    "ordinal_pretrain": "对比预训练＋有序分类",
    "ordinal_pretrain_tuned": "组合方法，内部选择学习率",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decision(row):
    p = row["probabilities"]
    return None if p is None else int(p[1] + p[2] > .5) + int(p[2] > .5)


def qualifies(candidate, baseline):
    more_correct = (candidate["correct_count"] > baseline["correct_count"]
                    and candidate["severe_to_mild_count"] <= baseline["severe_to_mild_count"]
                    and candidate["two_level_error_count"] <= baseline["two_level_error_count"])
    fewer_severe_errors = (candidate["severe_to_mild_count"] <= baseline["severe_to_mild_count"] - 1
                          and candidate["correct_count"] >= baseline["correct_count"] - 2
                          and candidate["two_level_error_count"] <= baseline["two_level_error_count"])
    return dict(passed=more_correct or fewer_severe_errors, more_correct_without_more_extreme_errors=more_correct,
                fewer_severe_to_mild_without_excess_correct_loss=fewer_severe_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.root / "summary.json")
    audit = read(args.root / "independent-audit.json")
    assert audit["passed"] and audit["variants_checked"] == 18
    baseline = summary["balanced_reference"]["means"]
    decisions = {name: qualifies(item["means"], baseline) for name, item in summary.items() if name != "balanced_reference"}
    paired = {}
    for method in ("ordinal_pretrain", "ordinal_pretrain_tuned"):
        paired[method] = []
        for seed in (42, 43, 44):
            control = read(args.root / "runs" / f"balanced_reference_seed{seed}/oof_results.json")
            result = read(args.root / "runs" / f"{method}_seed{seed}/oof_results.json")
            before = {r["sample_id"]: r for r in control["records"]}
            counts = dict(seed=seed, newly_correct=0, newly_wrong=0, severe_to_mild_changed_to_moderate=0,
                          severe_to_mild_changed_to_correct_severe=0, newly_severe_to_mild=0)
            for row in result["records"]:
                previous = before[row["sample_id"]]
                a, b, y = decision(previous), decision(row), row["label"]
                counts["newly_correct"] += int(a != y and b == y)
                counts["newly_wrong"] += int(a == y and b != y)
                if y == 2:
                    counts["severe_to_mild_changed_to_moderate"] += int(a == 0 and b == 1)
                    counts["severe_to_mild_changed_to_correct_severe"] += int(a == 0 and b == 2)
                    counts["newly_severe_to_mild"] += int(a != 0 and b == 0)
            assert counts["newly_correct"] - counts["newly_wrong"] == result["metrics"]["correct_count"] - control["metrics"]["correct_count"]
            paired[method].append(counts)
    lr_counts = {str(seed): {"0.0001": 0, "0.00005": 0} for seed in (42, 43, 44)}
    for seed in (42, 43, 44):
        result = read(args.root / "runs" / f"ordinal_pretrain_tuned_seed{seed}/oof_results.json")
        for choice in result["choices"]:
            lr_counts[str(seed)]["0.0001" if choice["lr"] == 1e-4 else "0.00005"] += 1
    public = dict(results=summary, continuation_conditions=decisions, paired_changes=paired,
                  inner_selected_learning_rate_counts=lr_counts,
                  independent_audit={k: v for k, v in audit.items() if k != "checks"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")

    def percentage(value):
        return f"{100 * value:.2f}%"

    def sequence(name, key):
        return "、".join(str(r["metrics"][key]) for r in summary[name]["per_seed"])

    def accuracy_sequence(name):
        return "、".join(percentage(r["metrics"]["accuracy_all_assigned"]) for r in summary[name]["per_seed"])

    def recall(name, y):
        return statistics.mean(r["metrics"]["class_recall_all_assigned"][y] for r in summary[name]["per_seed"])

    main_fixed = summary["ordinal_pretrain"]["means"]
    tuned = summary["ordinal_pretrain_tuned"]["means"]
    lines = ["# 10月16日：对比预训练＋有序分类，实际结果", "",
        "已完成三套随机种子、固定五折比较及训练资料内部的学习率选择，独立核查已通过。评价使用同一117份病例资料；5份全缺失资料拒判，仍计入准确率分母。资料份数不等于独立患者人数。", "",
        "## 先看准确率", "",
        f"当前对照平均准确率为{percentage(baseline['accuracy_all_assigned'])}；对比预训练＋有序分类为{percentage(main_fixed['accuracy_all_assigned'])}，提高{100*(main_fixed['accuracy_all_assigned']-baseline['accuracy_all_assigned']):.2f}个百分点。调整学习率后仍为{percentage(tuned['accuracy_all_assigned'])}。", "",
        "| 方法 | 三次准确率 | 平均准确率 |", "|---|---|---:|"]
    for name in ("balanced_reference", "ordinal_pretrain", "ordinal_pretrain_tuned"):
        lines.append(f"| {LABELS[name]} | {accuracy_sequence(name)} | {percentage(summary[name]['means']['accuracy_all_assigned'])} |")
    lines += ["", "三次顺序均为以42、43、44为基础的随机种子；病例分组相同。", "",
        f"**整体准确率提高，但中度识别明显下降。** 中度召回率，也就是中度被正确识别出来的比例，从{percentage(recall('balanced_reference', 1))}降至{percentage(recall('ordinal_pretrain', 1))}。重度判成正常/轻度的比例，从{percentage(baseline['severe_to_mild_count']/31)}升至{percentage(main_fixed['severe_to_mild_count']/31)}。", "",
        "固定组合与学习率调优后的组合都没有达到事先确定的继续条件，暂不替换APP模型。", "",
        "![各等级正确识别比例的比较](figures/contrastive-ordinal-class-correct.png)", "",
        "## 两项改动各自的结果", "",
        "以下四组均不加类别权重，只改变是否预训练、是否采用有序分类。上表当前对照另有类别平衡权重，所以保留这些组，分清改动的影响。", "",
        "| 方法 | 三次准确率 | 平均准确率 |", "|---|---|---:|"]
    for name in ("softmax", "softmax_pretrain", "ordinal", "ordinal_pretrain"):
        lines.append(f"| {LABELS[name]} | {accuracy_sequence(name)} | {percentage(summary[name]['means']['accuracy_all_assigned'])} |")
    lines += ["", "## 各等级能正确识别多少比例", "",
        "下表为三次平均召回率，即真实属于该等级的资料中，被正确识别出来的比例。拒判也计入各等级分母。", "",
        "| 方法 | 正常/轻度 | 中度 | 重度 | 平均分级误差 |", "|---|---:|---:|---:|---:|"]
    for name in ("balanced_reference", "ordinal_pretrain", "ordinal_pretrain_tuned"):
        rates = [percentage(recall(name, y)) for y in range(3)]
        lines.append(f"| {LABELS[name]} | {' | '.join(rates)} | {summary[name]['means']['mae']:.3f} |")
    lines += ["", "平均分级误差只计算112份有预测的资料；拒判没有定义等级距离。类别原始数量和各类错误数量放在后面的折叠内容，供核对。", "",
        "## 学习率怎样选择", "",
        "仅比较分级训练学习率0.0001与0.00005。每折先在自己的训练资料内部比较准确率、三类平均F1，再选学习率和轮数；外层留出资料没有参与选择。两个候选共享同一份预训练参数和随机起点。", "",
        "| 种子基数 | 五折中选0.0001的次数 | 选0.00005的次数 |", "|---|---:|---:|"]
    for seed, counts in lr_counts.items():
        lines.append(f"| {seed} | {counts['0.0001']} | {counts['0.00005']} |")
    lines += ["", "## 结论和接下来怎么做", "",
        "本轮已把用户指定的组合及有限学习率比较完整做完。固定组合和内部选学习率后的组合均未达到继续条件：准确率虽然提高，但重度判成正常/轻度以及两端互相错判增加。当前APP继续使用原冻结模型。", "",
        "学习率比较没有提高三次平均准确率，也没有解决中度识别差的问题。本轮停止继续扩大参数搜索。若后续仍研究这套组合，应先检查中度被判到哪一类，以及有序分类的训练目标如何影响三类的识别，而不是继续挑一次最好成绩。这个结果不等于证明所有对比预训练或有序分类都无效。", "",
        "旧六模态方法的71.43%已在原21份资料复现。本页使用当前五模态和117份五折评价，不能把两者直接相减归因于DINO或某一种损失。", "",
        "<details><summary>查看做法、核查与证据位置</summary>", "",
        "- 冻结Zenodo-DINOv2编码器，518图片处理、1536维图片特征、五类输入、DNA比例与阈值双输入、128维融合、0.15资料随机隐藏及常规增强均保持不变。",
        "- 多模态预训练30轮、批量32、学习率0.0001、温度0.1；只用训练资料中实际同时存在的模态对。保存了每轮配对数、有效更新次数、损失及预训练权重。它只预训练模态编码和投影，不预训练Transformer融合。",
        "- 分级训练批量8、最多50轮、连续5轮内部验证不提高则停止。固定比较学习率0.0001，有序头用原有两个累计事件的二分类损失，不叠加距离损失。",
        "- 所有组重新训练，没有因旧实验名称相同就跳过训练。当前平衡对照的三种子，以及不加权三分类和有序分类的首种子，都与旧归档逐份核对。",
        f"- 独立CPU程序重新组装输入、严格加载权重并重算预测：{audit['variants_checked']}组三种子结果、{audit['selected_outer_classifiers']}次选中外层模型，重算{audit['replayed_probability_vectors']}组概率；最大绝对误差{audit['maximum_probability_error']:.3g}，小于1e-4。同时核对内部轮数、学习率选择、训练词表、分组及预训练记录。",
        "- 源码：scripts/research/run_contrastive_ordinal_cv.py、audit_contrastive_ordinal_cv.py、summarize_contrastive_ordinal_cv.py；源码快照与SHA另存，未用Git HEAD冒充未提交代码。",
        "- 本地私有证据：outputs/experiments/sysu/contrastive-ordinal-20261016/attempt1/。远程完整输出：/home/zjkj/24334067/OPMD-Pred-SYSU-Proxy-Balanced/outputs/sysu_contrastive_ordinal_20261016_attempt1。",
        "- 本地恢复包包含源码、输入表和划分、逐份预测、训练记录以及172份阶段分类器。共享初始化/预训练权重initial.pt和优化器续跑文件last.pt完整保留服务器，SHA列在清单中；没有声称它们也在本地传输包内。",
        "- 旧22份没有重新预测，原病例、旧71.43%权重与APP冻结模型未覆盖。逐份预测、源数据及模型不放公开报告。",
        "- 类别分母为正常/轻度55份、中度31份、重度31份；5份拒判含3份正常/轻度及2份中度。重度判成正常/轻度的比例以31份真实重度资料为分母。",
        "- [完整汇总数字](contrastive-ordinal-results.json)、[执行前计划](contrastive-ordinal-plan.md)、[此前实验及旧成绩复现](../2026-10-15/model-comparison-results/README.md)。", "", "</details>", ""]
    lines += ["<details><summary>原始判对数量与错误数量，仅供核对</summary>", "",
        "| 方法 | 三次判对／117 | 重度判成正常/轻度 | 两端等级互相错判 |", "|---|---|---|---|"]
    for name in LABELS:
        lines.append(f"| {LABELS[name]} | {sequence(name, 'correct_count')} | {sequence(name, 'severe_to_mild_count')} | {sequence(name, 'two_level_error_count')} |")
    for name in ("ordinal_pretrain", "ordinal_pretrain_tuned"):
        m = summary[name]["means"]
        lines += ["", f"{LABELS[name]}：平均正确数变化{m['correct_count']-baseline['correct_count']:+.2f}份；重度判成正常/轻度平均{m['severe_to_mild_count']:.2f}份（对照{baseline['severe_to_mild_count']:.2f}份）；两端等级互相错判平均{m['two_level_error_count']:.2f}份（对照{baseline['two_level_error_count']:.2f}份）。"]
    lines += ["", "</details>", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(report=str(args.output), fixed_combination_passed=decisions["ordinal_pretrain"]["passed"],
                         tuned_combination_passed=decisions["ordinal_pretrain_tuned"]["passed"])))


if __name__ == "__main__":
    main()
