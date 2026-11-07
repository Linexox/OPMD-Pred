"""Experimental adapter for an injected, already verified binary predictor.

This adapter is not selected by the running service.  It preserves the old
three-class engine until a private binary model and target environment are
explicitly configured and verified.
"""
from pathlib import Path
import hashlib
import json
import math

import binary_inputs
from binary_inputs import binary_fields_from_visit
from core import HISTORY, digest
from grade_reference import grade_reference


def binary_input_fingerprints(fields, photo_sha):
    """Record effective inputs without saving the raw report values a second time."""
    groups = {
        'history': {key: fields[key] for key, _, _ in HISTORY},
        'tct': fields['TCT'],
        'dna': fields['dna_report_abnormal_ratio'],
        'methylation': fields['methylation_total'],
    }
    states = {'image': {'present': bool(photo_sha), 'sha256': photo_sha or hashlib.sha256(b'null').hexdigest(),
                        'source': 'verified_original_image'}}
    for name, value in groups.items():
        present = any(v is not None for v in value.values()) if isinstance(value, dict) else value is not None
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        states[name] = {'present': present, 'sha256': hashlib.sha256(encoded).hexdigest(), 'source': 'effective_binary_input'}
    return {'format': 'model-input-sha-v1', 'modalities': states}


class BinaryResearchEngine:
    requires_binary_client = True
    supports_dna_report_unusable = True

    def __init__(self, predictor, model_id):
        if not model_id:
            raise ValueError('二分类模型身份未配置')
        self.predictor = predictor
        self.model_id = model_id

    def predict(self, visit, root):
        fields = binary_fields_from_visit(visit.get('payload') or {})
        root = Path(root).resolve()
        relative_photo = visit.get('original_image') or visit.get('image')
        photo = None
        image_sha = ''
        if relative_photo:
            photo = (root / relative_photo).resolve()
            if not photo.is_relative_to(root) or not photo.is_file():
                raise ValueError('照片路径不可用，请重新保存资料')
            image_sha = visit.get('original_sha') or visit.get('image_sha') or ''
            if not image_sha or digest(photo) != image_sha:
                raise ValueError('照片已改变，请重新保存资料')
        cell_paths = []
        for relative in visit.get('cell_images', []) or []:
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError('细胞显微图路径不可用，请重新保存资料')
            cell_paths.append(candidate)
        genes = visit.get('methylation_genes') or {}
        if hasattr(self.predictor, 'supports_cell_images'):
            output = self.predictor.predict(fields, photo, cell_images=cell_paths, methylation_genes=genes)
        else:
            output = self.predictor.predict(fields, photo)
        common = {'input_revision':visit['revision'], 'model_id':self.model_id,
                  'image_sha256':image_sha, 'class_labels':['正常/轻度', '中度及以上']}
        fingerprint = getattr(self.predictor, 'verified_model_fingerprint', None)
        if fingerprint:
            parts = [fingerprint, digest(__file__), digest(binary_inputs.__file__)]
            common['model_fingerprint'] = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        if output.get('status') == 'no_input':
            return {**common, 'status':'abstain', 'probabilities':None,
                    'reason':'尚无可分析资料，请补充照片或检测信息'}
        probabilities = output.get('probabilities')
        if output.get('status') != 'predicted' or not isinstance(probabilities, (list,tuple)) or len(probabilities) != 2:
            raise ValueError('二分类模型输出格式不正确')
        probabilities = [float(value) for value in probabilities]
        if any(not math.isfinite(x) or x < 0 or x > 1 for x in probabilities) or abs(sum(probabilities)-1) > 1e-5:
            raise ValueError('二分类模型输出分数无效')
        prediction = int(probabilities[1] >= .5)
        if output.get('prediction') != prediction:
            raise ValueError('二分类模型类别与分数不一致')
        present = {'image':photo is not None,
                   'cell_images':bool(cell_paths),
                   'history':any(fields[k] is not None for k, _, _ in HISTORY),
                   'tct':fields['TCT'] is not None,
                   'dna':fields['dna_report_abnormal_ratio'] is not None,
                   'methylation':fields['methylation_total'] is not None or any(v is not None for v in genes.values())}
        vocabularies = getattr(self.predictor, 'tct_reference_vocabularies', None)
        tct_status = 'missing' if fields['TCT'] is None else 'unavailable' if vocabularies is None else (
            'known' if all(fields['TCT'] in values for values in vocabularies) else 'unknown')
        return {**common, 'status':'predicted', 'probabilities':probabilities,
                'grade_reference':getattr(self.predictor, 'grade_reference', None) or grade_reference(self.model_id),
                'median_class':common['class_labels'][prediction], 'present':present,
                **({'dna_excluded_reason':'report_unusable'} if (visit.get('payload') or {}).get('dna_report_unusable') is True else {}),
                'tct_check_status':tct_status,
                **({'unknown_tct':tct_status == 'unknown'} if tct_status in ('known','unknown') else {}),
                'model_inputs':{**binary_input_fingerprints(fields, image_sha), 'cell_images':{'present':bool(cell_paths),'count':len(cell_paths),'sha256': [digest(p) for p in cell_paths]}, 'methylation_genes':genes},
                'input_context':{'visit_id':visit['id'], 'patient_id':visit.get('patient_id'),
                    'site':visit.get('site'), 'visit_date':visit.get('visit_date'),
                    'task_id':visit.get('task_id'), 'revision':visit['revision']}}
