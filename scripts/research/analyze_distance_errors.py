"""Describe archived errors; never infer causality or evaluate test22."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--archive', required=True, type=Path); p.add_argument('--error-csv', required=True, type=Path); p.add_argument('--output', required=True, type=Path); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(a.error_csv.open(encoding='utf-8-sig')))
    control = {r['sample_id']: r for r in rows if r['seed']=='43' and r['method']=='control' and r['label']=='2'}
    candidate = {r['sample_id']: r for r in rows if r['seed']=='43' and r['method']=='distance05' and r['label']=='2'}
    new = sorted(candidate.keys() - control.keys())
    predictions = {}
    for method in ('control', 'distance05'):
        predictions[method] = {}
        for file in (a.archive / 'seed43' / method).rglob('outer_predictions.json'):
            for row in json.loads(file.read_text(encoding='utf-8'))['records']:
                predictions[method][row['sample_id']] = dict(row, fold=int(next(part[4:] for part in file.parts if part.startswith('fold') and part[4:].isdigit())))
    detail = []
    for sid in new:
        row = dict(candidate[sid])
        for method in ('control','distance05'):
            pred = predictions[method][sid]
            row[method + '_fold'] = pred['fold']
            row[method + '_probabilities'] = pred['probabilities']
        row['report_confirmation'] = '沿用10月9日PDF核对开发字段；不是患者端新勾选的确认'
        detail.append(row)
    result = {'date':'2026-10-11','source':'2026-10-10 archived paired predictions','seed':43,
              'control_severe_to_mild':len(control),'candidate_severe_to_mild':len(candidate),'new_count':len(new),
              'fixed_count':len(control.keys()-candidate.keys()),'retained_count':len(control.keys()&candidate.keys()),
              'new_cases':detail,'observation':'新增4条中，2条只有TCT和DNA，2条五模态齐全；不能将恶化全部归因于模态缺失。',
              'known_issue_flags':'这4条归档known_local_issue均为False；这不证明报告或标签不存在其他疑点。',
              'conclusion':'概率变化与错误增加相关，现有结果不能确认原因。距离损失仍未达到严重漏判减少门槛。',
              'next_experiment':'优先设计有方向的重度漏判代价对照；沿用分组开发交叉验证和正确率最多下降2个百分点标准，先固定一组代价，不进行大规模搜索。',
              'new_training':False,'old_test22_evaluated':False,'causal_claim':False}
    (a.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('new_count','fixed_count','retained_count','observation','new_training')},ensure_ascii=False))

if __name__=='__main__':main()
