"""Private research-only inference; no HTTP listener, training, or saved test answers."""
from pathlib import Path
import time
import importlib.metadata
import hashlib
import json
import inspect
from contextlib import nullcontext
from functools import wraps
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tabpfn.model_loading import load_fitted_tabpfn_model
from opmd_pred.model.vit import ImageEncoder
from opmd_pred.model.two_branch_fusion import TwoBranchFusion
from opmd_pred.data.transforms import make_image_transform
from sysu_binary_common import read,sha
from sysu_tabular_common import offline_predictions
from binary_tct_vocabulary import load_verified_tct_vocabularies

class PrivateBinaryPredictor:
    """Accept already source-checked numeric report fields and an optional photo."""
    def __init__(self, manifest, device='cuda', expected_seed=42, expected_fold='fold0'):
        start=time.perf_counter();self.manifest=read(manifest);m=self.manifest
        if m.get('path_base') == 'manifest':
            base = Path(manifest).resolve().parent
            resolve = lambda value: str((base / value).resolve()) if not Path(value).is_absolute() else value
            for key in ('encoder', 'fusion', 'tct_vocabulary'):
                if m.get(key): m[key] = resolve(m[key])
            for key in ('table_reference_models', 'identity_extra_sources'):
                m[key] = [resolve(value) for value in m.get(key, [])]
            m['sources_sha256'] = {resolve(key): value for key, value in m['sources_sha256'].items()}
        if m['seed'] != expected_seed or m['fold_directory'] != expected_fold:
            raise ValueError('Model identity differs from requested seed and fold')
        assert importlib.metadata.version('tabpfn')==m['tabpfn_version']
        self.columns=m['columns'];self.device=device;self.timings={}
        t=time.perf_counter()
        for p,v in m['sources_sha256'].items():
            if sha(p)!=v:raise ValueError('Private model source fingerprint differs')
        # The base TabPFN weights are loaded separately from fitted references.
        # Only expose an identity when those weights are explicitly listed and verified.
        extra=m.get('identity_extra_sources', [])
        vocab_path=m.get('tct_vocabulary')
        self.tct_reference_vocabularies=None
        if vocab_path:
            expected=m['sources_sha256'].get(vocab_path)
            reference_shas=[m['sources_sha256'].get(path) for path in m['table_reference_models']]
            if not expected or any(not value for value in reference_shas):
                raise ValueError('TCT词表缺少校验信息')
            self.tct_reference_vocabularies=load_verified_tct_vocabularies(vocab_path,expected,reference_shas)
        required=[m['encoder'],m['fusion'],*m['table_reference_models'],*extra,*([vocab_path] if vocab_path else [])]
        self.verified_model_fingerprint=None
        if extra and all(p in m['sources_sha256'] for p in required):
            from opmd_pred.model.resnet import ResNet18Encoder
            implementations=[__file__, inspect.getfile(ImageEncoder), inspect.getfile(ResNet18Encoder),
                             inspect.getfile(TwoBranchFusion), inspect.getfile(make_image_transform)]
            if vocab_path:
                implementations.append(inspect.getfile(load_verified_tct_vocabularies))
            identity={'artifacts':[m['sources_sha256'][p] for p in required],
                      'columns':self.columns,'device':str(self.device),
                      'torch':torch.__version__,'tabpfn':importlib.metadata.version('tabpfn'),
                      'dependencies':{name:importlib.metadata.version(name) for name in
                                      ('numpy','pandas','scikit-learn','torchvision','Pillow','skrub','joblib')},
                      'implementations':[sha(p) for p in implementations],
                      'inference_schema':'binary-resnet18-tabpfn-transformer-v1'}
            self.verified_model_fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.timings['source_hash_verification_seconds']=time.perf_counter()-t
        t=time.perf_counter();self.encoder=ImageEncoder(False,'resnet18').to(self.device)
        self.encoder.load_artifact(m['encoder']);self.encoder.freeze();self._synchronize()
        self.timings['encoder_load_seconds']=time.perf_counter()-t
        t=time.perf_counter();self.references=[]
        relocated = nullcontext()
        if m.get('path_base') == 'manifest':
            from tabpfn.classifier import TabPFNClassifier
            checkpoint = m['identity_extra_sources'][0]
            original_init = TabPFNClassifier.__init__
            @wraps(original_init)
            def relocated_init(instance, *args, **kwargs):
                kwargs['model_path'] = checkpoint
                return original_init(instance, *args, **kwargs)
            relocated = patch.object(TabPFNClassifier, '__init__', relocated_init)
        with offline_predictions(), relocated:
            for p in m['table_reference_models']:
                model=load_fitted_tabpfn_model(p,device=self.device)
                if m.get('path_base') == 'manifest': model.model_path = checkpoint
                if list(model.classes_)!=[0,1]:raise ValueError('Unexpected table class order')
                self.references.append(model)
        self._synchronize();self.timings['five_reference_load_seconds']=time.perf_counter()-t
        t=time.perf_counter();artifact=torch.load(m['fusion'],map_location='cpu',weights_only=True)
        self.fusion=TwoBranchFusion(**artifact['model_config']).to(self.device).eval()
        self.fusion.load_state_dict(artifact['state'],strict=True);self.transform=make_image_transform(False,'resnet18')
        self._synchronize();self.timings['fusion_load_seconds']=time.perf_counter()-t
        self.timings['total_load_seconds']=time.perf_counter()-start

    def _synchronize(self):
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict(self, fields=None, photo=None, *, query_repeats=24):
        self._synchronize();start=time.perf_counter();fields={} if fields is None else fields
        if query_repeats not in (1, 24):
            raise ValueError('Only validated reference batch sizes 1 and 24 are supported')
        unexpected=set(fields)-set(self.columns)
        if unexpected:raise ValueError('Unexpected clinical input keys')
        values={};mask=[]
        for col in self.columns:
            raw=fields.get(col)
            if raw is None or isinstance(raw,str) and not raw.strip():value=np.nan
            elif col=='TCT':value=str(raw).strip()
            else:
                value=float(raw)
                if not np.isfinite(value):raise ValueError('Numeric report field must be finite or None')
            values[col]=value;mask.append(float(pd.notna(value)))
        table_present=bool(any(mask));image_present=photo is not None
        if not table_present and not image_present:
            return {'status':'no_input','probabilities':None,'prediction':None,'seconds':time.perf_counter()-start}
        frame=pd.DataFrame([values],columns=self.columns);frame['TCT']=frame['TCT'].astype(object)
        table=np.zeros((1,2+len(self.columns)),np.float32);table_probabilities=None
        t=time.perf_counter()
        if table_present:
            # Match the batch-shaped GPU path used in the original study. Each
            # row is the *same* record; no other patient's data is introduced.
            repeated_frame = pd.concat([frame] * query_repeats, ignore_index=True)
            with offline_predictions():
                outputs=[np.asarray(model.predict_proba(repeated_frame),dtype=np.float64)[0] for model in self.references]
            table_probabilities=np.mean(outputs,axis=0)
            if not np.isfinite(table_probabilities).all():raise ValueError('Nonfinite table output')
            table[0]=list(table_probabilities)+mask
        self._synchronize();table_time=time.perf_counter()-t;t=time.perf_counter()
        image=torch.zeros((1,512),device=self.device)
        if image_present:
            with Image.open(photo) as opened:pixel=self.transform(opened.convert('RGB'))
            image=self.encoder(pixel[None].to(self.device))
        self._synchronize();photo_time=time.perf_counter()-t;t=time.perf_counter()
        batch={'image':image,'table':torch.tensor(table,device=self.device),
               'image_present':torch.tensor([image_present],device=self.device),'table_present':torch.tensor([table_present],device=self.device)}
        probabilities=self.fusion(batch).softmax(1)[0].cpu().numpy();self._synchronize()
        return {'status':'predicted','probabilities':probabilities.tolist(),'prediction':int(probabilities[1]>=.5),
                'table_probabilities':None if table_probabilities is None else table_probabilities.tolist(),
                'photo_feature':image[0].cpu().numpy().tolist(),'field_mask':mask,
                'photo_present':image_present,'table_present':table_present,
                'timings':{'table_seconds':table_time,'photo_seconds':photo_time,'fusion_seconds':time.perf_counter()-t,'total_seconds':time.perf_counter()-start}}
