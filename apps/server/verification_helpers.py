"""Convert archived development records for mobile-service verification."""
from core import HISTORY


def payload_from_csv(row):
    from opmd_pred.data.sysu import _number, _binary, MISSING
    names = ['是否吸烟', '吸烟年限（年）', '吸烟量（支/天）', '是否饮酒', '饮酒年限(年)',
             '饮酒量(ml/天)', '是否嚼槟榔', '嚼槟榔年限(年)', '嚼槟榔量（颗/天）']
    p = {key: (_binary(row[name]) if scale == 1 else _number(row[name]))
         for (key, _, scale), name in zip(HISTORY, names)}
    p.update(tct='' if row['TCT意见'] in MISSING else row['TCT意见'], methylation=_number(row['甲基化总数']),
             dna_count=_number(row['dna_abnormal_cells']), dna_total=_number(row['dna_analyzed_cells']),
             dna_threshold=_number(row['dna_threshold']), report_confirmed=True)
    return p
