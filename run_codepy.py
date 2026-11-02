#!/usr/bin/env python3
"""One session, frozen text/semantic models and certified CodePy execution.

Input is one JSON operation or an array of operations; operations are applied in
order. Examples and memory limits are in research/codepy/ssm_dispatch/PIPELINE.md.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from research.codepy.ssm_dispatch.pipeline import Pipeline

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--request',required=True,help='JSON file, or - for stdin')
    args=parser.parse_args()
    request=json.loads(sys.stdin.read() if args.request=='-' else Path(args.request).read_text())
    pipeline=Pipeline.load(args.session,seed=args.seed) if args.session.exists() else Pipeline(args.seed)
    results=[]
    for item in request if isinstance(request,list) else [request]:
        if item['operation']=='remember':
            pipeline.remember(item['entity'],item['property'],item['handle'],item['command'],np.asarray(item['arguments'],np.float32))
            results.append({'operation':'remember','handle':item['handle'],'stored':True})
        elif item['operation']=='develop':
            common=dict(entity=item['entity'],property=item['property'],handle=item['handle'],values=np.asarray(item['arguments'],np.float32),populations=item.get('populations',95))
            if any(k in item for k in ('X_train','Y_train','X_holdout','Y_holdout')):
                pairs={key:np.asarray(item[key],np.float32) for key in ('X_train','Y_train','X_holdout','Y_holdout')}
                result=pipeline.develop(task=item.get('task'),max_calls=item.get('max_calls'),use_macros=item.get('use_macros',False),macro_commands=item.get('macro_commands'),max_pair_ops=item.get('max_pair_ops'),**pairs,**common)
            else:
                result=pipeline.develop(item['task'],np.asarray(item['examples'],np.float32),**common)
            results.append({'operation':'develop',**asdict(result)})
        elif item['operation']=='observe':
            pipeline.observe(item['text']);results.append({'operation':'observe','consumed':True})
        elif item['operation']=='execute':
            result=pipeline.execute(item['question'])
            results.append({'operation':'execute','handle':result.handle,'command':result.command,'arguments':result.arguments.tolist(),
                            'output':result.output.tolist(),'independent_agrees':result.independent_agrees,'next_text_token':result.next_text_token})
        else:
            raise ValueError('Unknown operation')
    pipeline.save(args.session)
    print(json.dumps({'version':pipeline.version,'results':results},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
