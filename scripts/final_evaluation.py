"""Explicit final protocol commands. All outputs remain inside the project."""
import argparse
import json
import os
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
for key,folder in (('TEMP','tmp'),('TMP','tmp'),('TORCH_HOME','torch')):
    path=root/'.cache'/folder
    path.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(path)
from graphshield.final_evaluation import freeze, predict, evaluate_final, load_protocol, verify_seal

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','predict','evaluate','verify'])
    parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    if args.command=='freeze':
        folder,protocol=freeze(args.run_name)
        print(json.dumps({'protocol':str(folder/'protocol.json'),'transaction_selected':protocol['transaction_selected'],'ring_selected':protocol['ring_selected']},indent=2))
    elif args.command=='predict':
        result=predict(args.run_name)
        print(json.dumps({'sealed':True,'seconds':result['seconds'],'paysim':result['paysim'],'rings':result['rings']},indent=2))
    elif args.command=='evaluate':
        result=evaluate_final(args.run_name)
        print(json.dumps({'complete':True,'transaction_selected':result['paysim']['selected_model'],'ring_selected':result['selected_ring_model']},indent=2))
    else:
        folder,_,_=load_protocol(args.run_name)
        seal=verify_seal(folder)
        from graphshield.final_evaluation import read,verified
        status=read(folder/'status.json')
        if status['status']=='complete':verified(folder/'results.json',status['results_sha256'])
        print(json.dumps({'checksums':'passed','status':status['status'],'sealed_files':len(seal['files'])},indent=2))
