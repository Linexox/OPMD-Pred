"""Assemble the currently deployed private binary model, verifying every byte."""
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
BUNDLE=PROJECT/'models/app/binary'
REMOTE='/home/nexox/workspace/OPMD_Pred'
MANIFEST=REMOTE+'/runtime/checks/binary-tct-20261101/identity-tct.private.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    m=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','lnx','cat '+MANIFEST]))
    BUNDLE.mkdir(parents=True,exist_ok=True)
    replacements={}
    source_keys=['encoder','fusion','tct_vocabulary']
    sources=[m[k] for k in source_keys]+m['table_reference_models']+m['identity_extra_sources']
    for original in sources:
        name=Path(original).name
        local=BUNDLE/name
        remote=original if original.startswith('/') else REMOTE+'/models/binary-research-20261031/code/'+original
        expected=m['sources_sha256'][original]
        if not local.exists():
            incoming=local.with_suffix(local.suffix+'.incoming')
            subprocess.run(['scp','-q','lnx:'+remote,str(incoming)],check=True)
            assert sha(incoming)==expected,name
            incoming.rename(local)
        assert sha(local)==expected,name
        replacements[original]=name
    converted=dict(m)
    converted['path_base']='manifest'
    for k in source_keys:converted[k]=replacements[m[k]]
    for k in ['table_reference_models','identity_extra_sources']:converted[k]=[replacements[p] for p in m[k]]
    converted['sources_sha256']={replacements[p]:m['sources_sha256'][p] for p in sources}
    (BUNDLE/'manifest.json').write_text(json.dumps(converted,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (BUNDLE/'README.md').write_text('''# 当前APP二分类部署模型

ResNet-18＋TabPFN＋Transformer，随机种子42、fold0。参数来自当前Linux部署，逐文件SHA核对一致；不是新训练模型。manifest.json中的路径相对于本目录。

encoder.pt是图片编码器；fusion.pt是融合模型；table-*.tabpfn_fit是5个表格参考模型；ckpt是TabPFN基础权重；词表JSON用于检查TCT写法。

表格参考模型包含训练上下文，整个目录仅供服务端使用，不提交Git，不进入APK和公开下载。旧三分类文件仍保留在上一层，供历史复现和本机旧服务使用。
''',encoding='utf8')
    print(json.dumps({'artifacts':len(sources),'sha_verified':True,'size_MB':round(sum((BUNDLE/Path(p).name).stat().st_size for p in sources)/1024**2,1)}))


if __name__=='__main__':main()
