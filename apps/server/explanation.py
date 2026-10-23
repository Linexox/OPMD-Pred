"""Explain frozen-model inputs and compare explicitly related saved analyses."""
import hashlib
import json
import math
from datetime import datetime

INPUT_LABELS = {'image': '照片', 'history': '病史', 'tct': 'TCT', 'dna': 'DNA', 'methylation': '甲基化'}
CLASS_LABELS = ('正常/轻度', '中度', '重度')


def input_fingerprints(batch, present):
    """Identify effective model inputs, including partial numeric masks and unknown TCT."""
    state = {}
    for name in INPUT_LABELS:
        available = bool(present[name].item())
        keys = ('image_features',) if name == 'image' else ('tct',) if name == 'tct' else (name, name + '_mask')
        values = {key: batch[key].detach().cpu().tolist() for key in keys} if available else None
        encoded = json.dumps(values, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
        state[name] = {'present': available, 'sha256': hashlib.sha256(encoded).hexdigest(),
                       'source': 'image_features' if name == 'image' else 'encoded_values_and_masks'}
    return {'format': 'model-input-sha-v1', 'modalities': state}


def explain_embeddings(model, embeddings, present, expected_probabilities):
    """Exact modality SHAP for three class scores and the actual two decision checks."""
    import torch
    from opmd_pred.analysis.shapley import interaction_values, shapley_values

    names = tuple(model.fusion.names)
    if names != tuple(INPUT_LABELS) or model.head_kind != 'softmax':
        raise ValueError('当前解释接口仅支持已绑定的五模态三分类模型')
    count = 1 << len(names)
    device = next(iter(embeddings.values())).device
    masks = torch.arange(count, device=device)
    coalition_present = {name: present[name].bool() & ((masks & (1 << i)) != 0) for i, name in enumerate(names)}
    expanded = {name: value.expand(count, -1) for name, value in embeddings.items()}
    with torch.inference_mode():
        probabilities = model.forward_from_embeddings(expanded, coalition_present).softmax(1)
    observed_mask = sum(1 << i for i, name in enumerate(names) if bool(present[name].item()))
    replay_error = float((probabilities[observed_mask] - probabilities.new_tensor(expected_probabilities)).abs().max())
    if not torch.isfinite(probabilities).all() or replay_error > 1e-4:
        raise ValueError('解释计算与本次预测未能对应，未显示解释')
    # Collapse identical effective inputs so naturally absent modalities stay exact dummies.
    values = probabilities[masks & observed_mask].to(dtype=torch.float64)
    # The displayed endpoint is the actual saved single-record prediction, including at .5.
    values[(masks & observed_mask) == observed_mask] = values.new_tensor(expected_probabilities)
    targets = []
    checks = []
    for index in range(5):
        scalar = values[:, index] if index < 3 else values[:, 1:].sum(1) if index == 3 else values[:, 2]
        phi = shapley_values(scalar[None], len(names))[0]
        residual = float(scalar[observed_mask] - scalar[0] - phi.sum())
        if abs(residual) > 1e-6:
            raise ValueError('解释贡献未能还原本次分数，未显示解释')
        contributions = [dict(modality=name, label=INPUT_LABELS[name], present=bool(present[name].item()), value=float(phi[i])) for i, name in enumerate(names)]
        target = dict(id=f'class_{index}' if index < 3 else 'above_mild' if index == 3 else 'above_moderate',
                      label=CLASS_LABELS[index] if index < 3 else '是否超过轻度' if index == 3 else '是否超过中度',
                      value=float(scalar[observed_mask]), reference_value=float(scalar[0]), contributions=contributions, residual=residual)
        if index < 3:
            interactions = interaction_values(scalar[None], len(names))[0]
            target['interactions'] = sorted([
                dict(first=first, second=second, value=float(interactions[i, j]))
                for i, first in enumerate(names) for j, second in enumerate(names) if j > i
                and bool(present[first].item()) and bool(present[second].item())
            ], key=lambda pair: abs(pair['value']), reverse=True)
            targets.append(target)
        else:
            checks.append(dict(target, threshold=.5, passed=target['value'] > .5))
    decision = sum(check['passed'] for check in checks)
    return dict(status='ready', format='modality-shap-v1', method='exact_missing_token_shap',
                reference='all_inputs_missing_model_reference', model_head='softmax', rule='fixed_median_0.5',
                default_target=f'class_{decision}', targets=targets, decision_checks=checks,
                coalition_count=count, replay_error=replay_error)


def run_summary(run, relation='same_visit_revision'):
    result = run['result']
    context = result.get('input_context', {})
    return dict(id=run['id'], revision=run['revision'], created=run['created'],
                visit_date=context.get('visit_date', ''), site=context.get('site', ''),
                probabilities=result.get('probabilities', []), median_class=result.get('median_class', ''), relation=relation)


def compare_saved_runs(current, previous, source_visit_id='', source_revision=None):
    """Compare one visit's revisions or a task's exact source revision; never arbitrary visits."""
    def unavailable(reason):
        return {'status': 'unavailable', 'reason': reason}

    after, before = current['result'], previous['result']
    if after.get('status') != 'predicted' or before.get('status') != 'predicted':
        return unavailable('两次分析均需成功输出分数，才能对照')
    ca, cb = after.get('input_context'), before.get('input_context')
    if not ca or not cb:
        return unavailable('旧分析没有保存完整的输入对应信息，请使用更新后的分析记录')
    if any(not ca.get(key) or not cb.get(key) for key in ('patient_id', 'visit_id', 'site', 'visit_date')):
        return unavailable('记录对应信息不完整，无法建立前后对照')
    if ca['visit_id'] != current['visit_id'] or cb['visit_id'] != previous['visit_id']:
        return unavailable('分析与就诊记录不对应')
    if ca.get('revision') != current['revision'] or cb.get('revision') != previous['revision']:
        return unavailable('分析与资料版本不对应')
    if ca['patient_id'] != cb['patient_id'] or ca['site'] != cb['site']:
        return unavailable('仅对照同一档案、同一病损部位的资料')
    relation = 'same_visit_revision'
    if current['visit_id'] == previous['visit_id']:
        if current['revision'] <= previous['revision'] or ca['visit_date'] != cb['visit_date']:
            return unavailable('请选择本次就诊较早的资料版本，日期或部位变更后不自动对照')
    elif previous['visit_id'] == source_visit_id and previous['revision'] == source_revision and ca.get('task_id'):
        relation = 'material_task_source'
    else:
        return unavailable('仅对照同次资料更新，或明确关联的补资料任务')
    if datetime.fromisoformat(previous['created']) > datetime.fromisoformat(current['created']):
        return unavailable('请选择较早完成的分析')
    for key in ('classifier_sha256', 'image_encoder_sha256', 'rule'):
        if not after.get(key) or after.get(key) != before.get(key):
            return unavailable('两次使用的模型或分级规则不同，不混算资料变化的影响')
    for result in (before, after):
        p = result.get('probabilities')
        if not isinstance(p, list) or len(p) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in p) or abs(sum(p) - 1) > 1e-4:
            return unavailable('历史分数不完整，无法对照')
    sa, sb = after.get('model_inputs', {}), before.get('model_inputs', {})
    if sa.get('format') != 'model-input-sha-v1' or sb.get('format') != sa['format']:
        return unavailable('这条旧分析未记录当时实际使用的资料，暂时无法对照')
    changes = []
    for name, label in INPUT_LABELS.items():
        a, b = sa.get('modalities', {}).get(name), sb.get('modalities', {}).get(name)
        if not a or not b or not a.get('sha256') or not b.get('sha256'):
            return unavailable('实际输入记录不完整，无法确认资料变化')
        if name == 'image' and a['present'] and b['present'] and a.get('source') != b.get('source'):
            return unavailable('两次照片的记录方式不同，暂时无法确认是否改变')
        kind = 'added' if a['present'] and not b['present'] else 'removed' if b['present'] and not a['present'] else 'changed' if a['present'] and a['sha256'] != b['sha256'] else 'unchanged'
        changes.append(dict(modality=name, label=label, kind=kind))
    explanation_changes = []
    ea, eb = after.get('explanation', {}), before.get('explanation', {})
    if ea.get('status') == eb.get('status') == 'ready' and ea.get('format') == eb.get('format') == 'modality-shap-v1':
        previous_targets = {target['id']: target for target in eb['targets']}
        for target in ea['targets']:
            old = previous_targets.get(target['id'])
            if old:
                old_values = {item['modality']: item['value'] for item in old['contributions']}
                explanation_changes.append(dict(target=target['id'], label=target['label'], contributions=[
                    dict(modality=item['modality'], label=item['label'], before=old_values[item['modality']], after=item['value'])
                    for item in target['contributions'] if item['modality'] in old_values
                ]))
    return dict(status='compared', current=run_summary(current, relation), previous=run_summary(previous, relation),
                probability_deltas=[a - b for a, b in zip(after['probabilities'], before['probabilities'])],
                changed_inputs=changes, explanation_changes=explanation_changes)
