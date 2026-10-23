"""Separate corrected classifications from merely less distant ordinal errors."""
import argparse
import json
from pathlib import Path


def decision(row):
    p = row["probabilities"]
    return None if p is None else int(p[1] + p[2] > .5) + int(p[2] > .5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    public, private = {}, {}
    for name in ("legacy_subset", "contrastive30", "vit_softmax", "dna_log_standardized", "legacy_pipeline"):
        public[name], private[name] = [], []
        for seed in (42, 43, 44):
            control_name = "dino_softmax_balanced" if seed == 42 else f"dino_softmax_balanced_seed{seed}"
            candidate_name = name if seed == 42 and name != "legacy_pipeline" else f"{name}_seed{seed}"
            baseline = json.loads((args.root / "runs" / control_name / "oof_results.json").read_text())
            candidate = json.loads((args.root / "runs" / candidate_name / "oof_results.json").read_text())
            old = {r["sample_id"]: r for r in baseline["records"]}
            counts = dict(seed=seed, wrong_to_correct=0, correct_to_wrong=0,
                          severe_to_mild_removed=0, severe_to_mild_added=0,
                          removed_became_correct_severe=0, removed_became_wrong_moderate=0,
                          class_support=[0, 0, 0], baseline_class_correct=[0, 0, 0], candidate_class_correct=[0, 0, 0])
            rows = []
            for row in candidate["records"]:
                before = old[row["sample_id"]]
                assert row["label"] == before["label"]
                y = row["label"]
                a, b = decision(before), decision(row)
                counts["class_support"][y] += 1
                counts["baseline_class_correct"][y] += int(a == y)
                counts["candidate_class_correct"][y] += int(b == y)
                counts["wrong_to_correct"] += int(a != y and b == y)
                counts["correct_to_wrong"] += int(a == y and b != y)
                removed = y == 2 and a == 0 and b != 0
                added = y == 2 and a != 0 and b == 0
                counts["severe_to_mild_removed"] += int(removed)
                counts["severe_to_mild_added"] += int(added)
                if removed:
                    counts["removed_became_correct_severe"] += int(b == 2)
                    counts["removed_became_wrong_moderate"] += int(b == 1)
                if a != b:
                    rows.append(dict(sample_id=row["sample_id"], label=y, baseline_prediction=a, candidate_prediction=b,
                                     wrong_to_correct=a != y and b == y, correct_to_wrong=a == y and b != y,
                                     removed_severe_to_mild=removed, added_severe_to_mild=added))
            assert counts["wrong_to_correct"] - counts["correct_to_wrong"] == candidate["metrics"]["correct_count"] - baseline["metrics"]["correct_count"]
            assert counts["severe_to_mild_removed"] - counts["severe_to_mild_added"] == baseline["metrics"]["severe_to_mild_count"] - candidate["metrics"]["severe_to_mild_count"]
            public[name].append(counts)
            private[name].append(dict(seed=seed, rows=rows))
    (args.root / "paired-error-changes.public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.root / "paired-error-changes.private.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public["legacy_subset"]), flush=True)


if __name__ == "__main__":
    main()
