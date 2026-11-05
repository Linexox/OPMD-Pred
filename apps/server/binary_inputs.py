"""Map existing mobile visit fields to the research binary model input schema.

This module does not select a model or serve predictions.  Report quality must
still be assessed from its source; numeric values alone cannot establish it.
"""
from core import HISTORY, validate_payload

TCT_MISSING = {'', 'nan', 'NAN', 'NA', 'N/A', '/', '#VALUE!', 'None', 'null', 'NULL'}


def binary_fields_from_visit(payload):
    """Return fields in the units used by the SYSU binary research model."""
    checked = validate_payload(payload)
    unusable = payload.get('dna_report_unusable')
    if unusable is not None and not isinstance(unusable, bool):
        raise ValueError('DNA报告不可用标记须为布尔值')
    result = {key: checked[key] for key, _, _ in HISTORY}
    raw_tct = payload.get('tct')
    tct = '' if raw_tct is None else str(raw_tct).strip()
    result['TCT'] = None if tct in TCT_MISSING else tct
    # Only an explicit report-quality indication excludes the measurement.
    # Keep the original fields in the stored visit; absence is not rejection.
    result['dna_report_abnormal_ratio'] = None if unusable is True else checked['dna_ratio']
    result['methylation_total'] = checked['methylation']
    return result
