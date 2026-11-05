"""Validate TCT vocabularies against the exact saved TabPFN references."""
import hashlib
import json
from pathlib import Path

def load_verified_tct_vocabularies(path, expected_file_sha, reference_shas):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_file_sha:
        raise ValueError('TCT参考词表校验失败')
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('format')!='binary-tct-reference-vocabulary-v1':
        raise ValueError('TCT参考词表格式错误')
    refs=data.get('references')
    if not isinstance(refs,list) or len(refs)!=5 or [r.get('model_sha256') for r in refs]!=reference_shas:
        raise ValueError('TCT参考词表与模型不匹配')
    output=[]
    for ref in refs:
        values=ref.get('tct_values')
        if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() or v!=v.strip() for v in values) or len(values)!=len(set(values)):
            raise ValueError('TCT参考词表内容错误')
        output.append(frozenset(values))
    return tuple(output)
