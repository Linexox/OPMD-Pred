"""Five complete private classifiers, combined with fixed equal probability weights."""
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from prototype_binary_predictor import PrivateBinaryPredictor
from sysu_binary_common import read, sha


class MeanEnsemblePredictor:
    def __init__(self, manifest, device='cpu'):
        started = time.perf_counter()
        path = Path(manifest).resolve(); self.manifest = read(path)
        m = self.manifest
        if m.get('format') != 'sysu-five-complete-model-mean-v1' or len(m.get('members', [])) != 5:
            raise ValueError('Expected five complete model manifests')
        if [x['weight'] for x in m['members']] != [.2] * 5 or m.get('classifier_dropout_during_training') != .1:
            raise ValueError('Expected the selected equal-weight ensemble')
        self.columns = m['columns']; self.weights = np.full(5, .2, dtype=np.float64)
        for name, fingerprint in m['sources_sha256'].items():
            file = (path.parent / name).resolve()
            if not file.is_relative_to(path.parent) or sha(file) != fingerprint:
                raise ValueError('Ensemble artifact checksum differs')
        self.members = []
        for entry in m['members']:
            child = (path.parent / entry['manifest']).resolve()
            if not child.is_relative_to(path.parent):
                raise ValueError('Member manifest outside ensemble package')
            predictor = PrivateBinaryPredictor(child, device=device, expected_seed=entry['seed'],
                                               expected_fold=entry['fold_directory'])
            if predictor.columns != self.columns or not predictor.verified_model_fingerprint:
                raise ValueError('Incomplete ensemble member identity')
            if self.members:
                # All five use the exact same frozen encoder; retain a single copy in memory.
                predictor.encoder = self.members[0].encoder
            self.members.append(predictor)
        self.tct_reference_vocabularies = tuple(v for p in self.members for v in p.tct_reference_vocabularies)
        self.grade_reference = m['grade_reference']
        identity = {'manifest_sha256': sha(path), 'members': [p.verified_model_fingerprint for p in self.members],
                    'weights': self.weights.tolist(), 'implementation_sha256': sha(inspect.getfile(type(self))),
                    'schema': 'five-complete-model-equal-mean-v1'}
        self.verified_model_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.timings = {'total_load_seconds': time.perf_counter() - started}

    def predict(self, fields=None, photo=None, *, query_repeats=24):
        started = time.perf_counter()
        outputs = [p.predict(fields, photo, query_repeats=query_repeats) for p in self.members]
        if all(o['status'] == 'no_input' for o in outputs):
            return {'status': 'no_input', 'probabilities': None, 'prediction': None,
                    'seconds': time.perf_counter() - started}
        if any(o['status'] != 'predicted' for o in outputs):
            raise ValueError('Ensemble members disagree about available inputs')
        probabilities = np.asarray([o['probabilities'] for o in outputs], dtype=np.float64)
        if probabilities.shape != (5, 2) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid member probabilities')
        mean = np.mean(probabilities, axis=0)
        if not np.allclose(probabilities.sum(1), 1., atol=1e-5, rtol=0):
            raise ValueError('Member probabilities do not sum to one')
        return {'status': 'predicted', 'probabilities': mean.tolist(), 'prediction': int(mean[1] >= .5),
                'member_probabilities': probabilities.tolist(), 'ensemble_weights': self.weights.tolist(),
                'photo_present': outputs[0]['photo_present'], 'table_present': outputs[0]['table_present'],
                'timings': {'total_seconds': time.perf_counter() - started,
                            'member_seconds': [o['timings']['total_seconds'] for o in outputs]}}
