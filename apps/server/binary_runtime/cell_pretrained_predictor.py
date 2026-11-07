"""Inference adapter for the migrated oral-cell pretrained research models."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/research'))
from run_methylation_cell_transfer import Fusion


class CellPretrainedPredictor:
    def __init__(self, root, device='cpu'):
        self.root = Path(root).resolve()
        self.device = torch.device(device)
        self.model_root = self.root / 'outputs/experiments/oral-cytology-transfer-20261106/sysu-attempt1/oral_pretrained'
        self.members = []
        for fold in range(5):
            path = self.model_root / f'fold{fold}/refit/classifier.pt'
            artifact = torch.load(path, map_location='cpu', weights_only=True)
            model = Fusion(artifact['table_size'], artifact['config']).to(self.device).eval()
            model.load_state_dict(artifact['state'], strict=True)
            self.members.append((model, artifact))
        self.photo = self._load_photo_encoder()
        self.cell = self._load_cell_encoder()
        self.requires_binary_client = True
        self.supports_dna_report_unusable = True
        self.supports_cell_images = True
        self.tct_reference_vocabularies = tuple(a['tct_vocab'] for _, a in self.members)
        self.grade_reference = {
            'method': 'new_model_original_grade_median',
            'reference_id': 'sysu-oral-cell-pretrained-seed42-fivefold-20261106',
            'score_index': 1,
            'records': 112,
            'scope': '新模型五折留出预测，按原始三等级分组的中位数',
            'anchors': [
                {'label': '正常/轻度', 'count': 52, 'score': 0.01969047449529171},
                {'label': '中度', 'count': 29, 'score': 0.9153531789779663},
                {'label': '重度', 'count': 31, 'score': 0.9882242679595947},
            ],
            'labels': ['正常/轻度', '中度及以上'],
        }
        self.model_id = 'sysu-oral-cell-pretrained-mean5-seed42'

    def _load_photo_encoder(self):
        from opmd_pred.model.resnet import ResNet18Encoder
        from opmd_pred.data.transforms import make_image_transform
        path = self.root / 'outputs/experiments/zenodo/first-step-supplement/classification/resnet18/balanced/seed42/attempt1/encoder.pt'
        encoder = ResNet18Encoder(False).to(self.device)
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        encoder.load_state_dict(artifact['state'], strict=True); encoder.eval(); encoder.requires_grad_(False)
        self.photo_transform = make_image_transform(False, 'resnet18')
        return encoder

    def _load_cell_encoder(self):
        path = self.root / 'outputs/experiments/oral-cytology-transfer-20261106/pretrain-attempt1/best.pt'
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        model = resnet18(weights=None); model.fc = nn.Identity(); state = artifact['model_state']
        state = {k: v for k, v in state.items() if not k.startswith('fc.')}
        model.load_state_dict(state, strict=False); model.to(self.device).eval(); model.requires_grad_(False)
        return model

    @staticmethod
    def _starts(length):
        last = max(0, length - 256)
        return sorted(set(list(range(0, last + 1, 128)) + [last]))

    @torch.no_grad()
    def _cell_feature(self, paths):
        features = []
        for path in paths:
            with Image.open(path) as opened:
                image = opened.convert('RGB')
                tiles = []
                for top in self._starts(image.height):
                    for left in self._starts(image.width):
                        tile = Image.new('RGB', (256, 256), 'white')
                        tile.paste(image.crop((left, top, min(left + 256, image.width), min(top + 256, image.height))), (0, 0))
                        tensor = TF.to_tensor(tile.resize((224, 224), Image.Resampling.BILINEAR))
                        tiles.append(TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225]))
            values = []
            for start in range(0, len(tiles), 32):
                values.append(self.cell(torch.stack(tiles[start:start + 32]).to(self.device)).cpu())
            features.append(torch.cat(values).mean(0))
        return torch.stack(features).mean(0).numpy() if features else np.zeros(512, dtype=np.float32)

    @staticmethod
    def _number(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def _batch(self, fields, genes, photo_feature, cell_feature, artifact):
        history = ['smoking', 'smoking_years', 'smoking_amount', 'alcohol', 'alcohol_years', 'alcohol_amount', 'betel', 'betel_years', 'betel_amount']
        numeric = np.asarray([self._number(fields.get(k)) for k in history] + [self._number(fields.get('dna_report_abnormal_ratio'))], dtype=np.float32)
        mean, scale = np.asarray(artifact['normalization_mean'], dtype=np.float32), np.asarray(artifact['normalization_scale'], dtype=np.float32)
        observed = np.isfinite(numeric); table = np.zeros((1, artifact['table_size']), dtype=np.float32)
        table[0, :10] = np.where(observed, np.clip((np.nan_to_num(numeric, nan=0.) - mean) / scale, -5, 5), 0)
        table[0, 10:20] = observed
        tct = str(fields.get('TCT') or '').strip()
        vocab = artifact['tct_vocab']
        if tct: table[0, 20 + (vocab.index(tct) + 1 if tct in vocab else 0)] = 1
        gene_values = np.asarray([self._number(genes.get(g)) for g in ['CDKN2A','TIMP3','ZNF582','TMEFF2','TFPI2','SHOX2','PAX1','MGMT','SOX1','DAPK1']], dtype=np.float32)
        gene_mask = np.isfinite(gene_values)
        methyl = np.concatenate([np.nan_to_num(gene_values).astype(np.float32), gene_mask.astype(np.float32)])[None]
        present = np.asarray([[photo_feature is not None, gene_mask.any(), observed.any() or bool(tct), bool(np.any(cell_feature))]], dtype=bool)
        return {'photo': torch.tensor(photo_feature[None] if photo_feature is not None else np.zeros((1, 512), dtype=np.float32), device=self.device),
                'methyl': torch.tensor(methyl, device=self.device), 'table': torch.tensor(table, device=self.device),
                'cell': torch.tensor(cell_feature[None], device=self.device), 'present': torch.tensor(present, device=self.device)}

    @torch.no_grad()
    def predict(self, fields=None, photo=None, *, cell_images=None, methylation_genes=None):
        started = time.perf_counter(); fields = fields or {}; genes = methylation_genes or {}
        photo_feature = None
        if photo is not None:
            with Image.open(photo) as opened: pixel = self.photo_transform(opened.convert('RGB'))
            photo_feature = self.photo(pixel[None].to(self.device)).cpu().numpy()[0]
        cell_feature = self._cell_feature(cell_images or [])
        if photo_feature is None and not genes and not cell_images and not any(v is not None for v in fields.values()):
            return {'status': 'no_input', 'probabilities': None, 'prediction': None}
        outputs = []
        for model, artifact in self.members:
            outputs.append(model(self._batch(fields, genes, photo_feature, cell_feature, artifact)).softmax(1)[0].cpu().numpy())
        probabilities = np.mean(outputs, axis=0)
        return {'status':'predicted','probabilities':probabilities.tolist(),'prediction':int(probabilities[1] >= .5),
                'class_labels':['正常/轻度','中度及以上'], 'present':{'image':photo_feature is not None,'cell_images':bool(cell_images),'history':any(fields.get(k) is not None for k in ('smoking','alcohol','betel')),'tct':bool(fields.get('TCT')),'dna':fields.get('dna_report_abnormal_ratio') is not None,'methylation':bool(genes)},
                'cell_image_count':len(cell_images or []),'member_probabilities':[x.tolist() for x in outputs],
                'ensemble_weights':[.2]*5,'timings':{'total_seconds':time.perf_counter()-started}}
