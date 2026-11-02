"""Generate through both the base network and the jointly trained pointer head."""
import argparse,json
from pathlib import Path
from runtime import load

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--prompt',default='memory is');ap.add_argument('--sample-steps',type=int,default=8);ap.add_argument('--output-json');args=ap.parse_args()
    if args.sample_steps<0:ap.error('--sample-steps must be nonnegative')
    params,cfg,vocab,n=load(args.checkpoint)
    import jax,jax.numpy as jnp,numpy as np
    from text_tokenization import tokenize_with_config,encode_tokens,detokenize_with_config
    tokens=tokenize_with_config(args.prompt,cfg['tokenizer_config']);ids=encode_tokens(tokens,{s:i for i,s in enumerate(vocab)}).tolist();ids=ids or [vocab.index('<bos>')];kind=cfg['model']
    ff=jax.jit(lambda p,x,s:n.forward(p,x,s,kind));last=None
    # Recompute short sample prefixes to preserve Transformer's training chunk policy.
    for _ in range(args.sample_steps):
        state=n.initial_model(params,1,kind,cfg['neural_context_copy']['capacity'])
        for start in range(0,len(ids),128):last,state=ff(params,jnp.asarray([ids[start:start+128]],jnp.int32),state)
        ids.append(int(jnp.argmax(last[0,-1])))
    result={'model':kind,'neural_pointer_used':True,'external_calibrators_enabled':False,'prompt':args.prompt,'sample':detokenize_with_config([vocab[i] for i in ids],cfg['tokenizer_config']),'sample_steps':args.sample_steps,'probability_sum':float(jnp.exp(last[0,-1]).sum()) if last is not None else None,'checkpoint':str(Path(args.checkpoint).resolve())}
    if args.output_json:Path(args.output_json).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
