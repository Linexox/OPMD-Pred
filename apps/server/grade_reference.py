"""Display-only cohort anchors; these never change the binary prediction."""
MODEL_ID = 'sysu-resnet18-tabpfn-transformer-seed42-fold0'


def grade_reference(model_id):
    if model_id != MODEL_ID:
        return None
    return {
        'method': 'original_grade_median',
        'reference_id': 'resnet18-tabpfn-transformer-seed42-oof-20261102',
        'score_index': 1,
        'records': 112,
        'scope': 'five-fold held-out development scores; pooled across folds',
        'anchors': [
            {'label': '正常/轻度', 'count': 52, 'score': 0.2859383523464203},
            {'label': '中度', 'count': 29, 'score': 0.5265828371047974},
            {'label': '重度', 'count': 31, 'score': 0.8265524506568909},
        ],
    }
