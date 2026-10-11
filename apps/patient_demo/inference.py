"""Pinned research inference, using the exact frozen-feature classifier interface."""
import json
import math
import os
import time
from pathlib import Path

import bootstrap  # noqa: F401
from core import HISTORY, digest, validate_payload

ENCODER_SHA = '58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65'
CLASSIFIER_SHA = 'f00181a650d1927c40cbc026b55b4f57dd4dd9946649cb775a85b60f2852a321'
LABELS = ('正常/轻度', '中度', '重度')


class ResearchEngine:
    def __init__(self, config):
        self.config = dict(config)
        self.model = None
        self.encoder = None
        self.fit = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        from opmd_pred.model.fusion import MultimodalOrdinalModel, NumericEncoder
        from opmd_pred.model.vit import ImageEncoder
        cpath, epath = Path(self.config['classifier']), Path(self.config['encoder'])
        if not cpath.is_file() or not epath.is_file():
            raise FileNotFoundError('模型文件未配置或不存在；没有生成替代结果')
        if digest(cpath) != CLASSIFIER_SHA or digest(epath) != ENCODER_SHA:
            raise ValueError('模型SHA不匹配，拒绝加载')
        saved = torch.load(cpath, map_location='cpu', weights_only=True)
        c = saved['config']
        expected = {'dimension': 128, 'image_encoder': 'dinov2_b14', 'head': 'softmax',
                    'fusion': 'transformer', 'dna_input_size': 2,
                    'modalities': ['image', 'history', 'tct', 'dna', 'methylation']}
        if saved.get('format') != 'sysu-frozen-feature-cv-classifier-v1' or any(c.get(k) != v for k, v in expected.items()):
            raise ValueError('当前分类器接口不匹配')
        if saved['image_encoder_sha256'] != ENCODER_SHA:
            raise ValueError('分类器与编码器绑定不匹配')
        if saved['stage_identity'].get('fold') != 0 or saved['stage_identity'].get('role') != 'refit':
            raise ValueError('仅支持预定fold0重拟合版本')
        model = MultimodalOrdinalModel(dimension=128, pretrained_image=False, tct_categories=c['tct_categories'],
                                      image_encoder='dinov2_b14', head='softmax', fusion_mode='transformer')
        model.fusion.dna_encoder = NumericEncoder(2, 128)
        model.fusion.freeze_image()
        model.fusion.image_encoder = None
        model.load_state_dict(saved['state'], strict=True)
        device = self.config.get('device', 'cpu')
        if device not in ('cpu', 'cuda') or device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('指定的推理设备不可用')
        self.device = torch.device(device)
        model.to(self.device).eval()
        encoder = ImageEncoder(pretrained=False, kind='dinov2_b14')
        encoder.load_artifact(epath)
        encoder.freeze()
        encoder.to(self.device).eval()
        self.model, self.encoder, self.fit = model, encoder, saved['fit']
        self.stage = saved['stage_identity']

    def _batch(self, payload, features, image_present):
        import torch
        from opmd_pred.data.sysu import _value_mask
        p = validate_payload(payload)
        history, hmask = _value_mask([p.get(k) for k, _, _ in HISTORY], [s for _, _, s in HISTORY])
        confirmed = p.get('report_confirmed', False)
        dna = p['dna_ratio'] if confirmed else None
        threshold = p['dna_threshold'] if dna is not None else None
        dna_values, dmask = _value_mask([dna, float(threshold == 2.5) if threshold is not None else None], [1, 1])
        methylation, mmask = _value_mask([p['methylation'] if confirmed else None], [10])
        tct = p['tct'] if confirmed else ''
        batch = {'history': history[None], 'history_mask': hmask[None], 'dna': dna_values[None],
                 'dna_mask': dmask[None], 'methylation': methylation[None], 'methylation_mask': mmask[None],
                 'tct': torch.tensor([self.fit['tct_vocab'].get(tct, 1) if tct else 0]),
                 'tct_present': torch.tensor([bool(tct)]), 'image_present': torch.tensor([image_present]),
                 'image_features': features}
        return {k: v.to(self.device) for k, v in batch.items()}

    def predict_features(self, payload, features, image_present=True):
        import torch
        self._load()
        batch = self._batch(payload, torch.as_tensor(features, dtype=torch.float32).reshape(1, 1536), image_present)
        fusion = self.model.fusion
        with torch.inference_mode():
            embeddings = {'image': fusion.image_projection(batch['image_features']), 'tct': fusion.tct_encoder(batch['tct'])}
            present = {'image': batch['image_present'], 'tct': batch['tct_present']}
            for name in ('history', 'dna', 'methylation'):
                embeddings[name] = getattr(fusion, name + '_encoder')(batch[name], batch[name + '_mask'])
                present[name] = batch[name + '_mask'].any(1)
            coverage = {k: bool(v.item()) for k, v in present.items()}
            unknown_tct = coverage['tct'] and int(batch['tct'][0]) == 1
            if not any(coverage.values()):
                return {'status': 'abstain', 'reason': '全部模型输入缺失或尚未确认', 'probabilities': None, 'present': coverage}
            probs = self.model.forward_from_embeddings(embeddings, present).softmax(1)[0].cpu().tolist()
            if not all(math.isfinite(v) for v in probs):
                raise ValueError('模型输出不是有限概率')
            decision = int(sum(probs[1:]) > .5) + int(probs[2] > .5)
            return {'status': 'predicted', 'probabilities': probs, 'median_class': LABELS[decision], 'present': coverage,
                    'unknown_tct': unknown_tct, 'tct_note': '使用unknown类别，建议核对原报告描述' if unknown_tct else '',
                    'classifier_sha256': CLASSIFIER_SHA, 'image_encoder_sha256': ENCODER_SHA,
                    'rule': 'fixed_median_0.5', 'stage_identity': self.stage,
                    'scope': 'SYSU三分类研究输出；未校准为癌变概率，未验证社区初筛或长期预测'}

    def predict(self, visit, root):
        from PIL import Image
        import torch
        from opmd_pred.data.transforms import make_image_transform
        start = time.monotonic()
        self._load()
        image_present = bool(visit.get('image'))
        report_paths = json.loads(visit.get('report') or '[]')
        if visit['payload'].get('report_confirmed') and [digest(Path(root) / p) for p in report_paths] != visit['payload'].get('report_sha256s', []):
            raise ValueError('已确认报告内容发生变化，请重新核对')
        features = torch.zeros(1, 1536)
        if image_present:
            path = Path(root) / (visit.get('original_image') or visit['image'])
            if digest(path) != (visit.get('original_sha') or visit['image_sha']):
                raise ValueError('本地照片已改变，请重新保存并提交')
            with Image.open(path) as image:
                image.load()
                tensor = make_image_transform(False, 'dinov2_b14')(image.convert('RGB'))[None].to(self.device)
            with torch.inference_mode():
                features = self.encoder(tensor)
        result = self.predict_features(visit['payload'], features, image_present)
        result.update(elapsed_seconds=round(time.monotonic() - start, 3), input_revision=visit['revision'],
                      image_sha256=visit.get('original_sha') or visit.get('image_sha', ''),
                      report_sha256s=visit['payload'].get('report_sha256s', []), image_quality=visit.get('quality', {}))
        return result


def load_engine_config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('classifier', 'encoder'):
        p = Path(c[key])
        c[key] = str(p if p.is_absolute() else Path(path).resolve().parent / p)
    return c
