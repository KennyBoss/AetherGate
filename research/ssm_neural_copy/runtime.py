"""Load the complete model with its recorded, frozen neural implementation."""
import os,sys,json,hashlib
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    os.environ['JAX_PLATFORMS']='cpu'
    import numpy as np
    path=Path(path).resolve()
    directory=next((p for p in [path.parent,path.parent.parent] if (p/'protocol.json').is_file()),None)
    if directory is None:raise ValueError('Keep the checkpoint together with its study protocol and source folder')
    protocol=json.loads((directory/'protocol.json').read_text());source=directory/'source'
    for name,h in protocol['source_sha256'].items():
        if sha(source/name)!=h:raise ValueError('Frozen source hash mismatch: '+name)
    if sha(source/'neural_context_copy.py')!=protocol['pointer_sha256']:raise ValueError('Pointer source hash mismatch')
    sys.path.insert(0,str(source))
    import jax.numpy as jnp
    import neural_context_copy as n
    import train_ssm_text as c
    if sha(n.__file__)!=protocol['pointer_sha256'] or sha(c.__file__)!=protocol['source_sha256']['train_ssm_text.py']:raise ValueError('Run in a fresh process to use the checkpoint source snapshot')
    with np.load(path,allow_pickle=False) as d:
        cfg=json.loads(str(d['config_json'].item()));vocab=json.loads(str(d['vocab_json'].item()));params={k[6:]:jnp.asarray(d[k]) for k in d.files if k.startswith('param_')}
        if any(k.startswith('opt_') or k.startswith('stream_') for k in d.files):raise ValueError('Expected inference weights only')
    if cfg.get('inference_recipe')!='joint_neural_context_copy' or cfg.get('model') not in ['ssm','transformer']:raise ValueError('Not a complete neural-copy checkpoint')
    if cfg['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('Protocol mismatch')
    from causal_frequency import frequency_kwargs
    if frequency_kwargs(cfg,len(vocab))!={}:raise ValueError('External calibrators are not allowed')
    head=cfg['neural_context_copy']
    if not head['jointly_trained'] or head['capacity']!=protocol['pointer']['capacity'] or params['pointer_current_w'].shape[1]!=head['dimension']:raise ValueError('Neural head configuration mismatch')
    if len(vocab)!=params['token_embed'].shape[0] or not all(np.isfinite(x).all() for x in params.values()):raise ValueError('Invalid model arrays')
    return params,cfg,vocab,n
