"""Installed-wheel and checkout launcher for the single-worker local application."""
import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

from .paths import ROOT


def main():
    for key,folder in (('TEMP','tmp'),('TMP','tmp'),('TORCH_HOME','torch')):
        path=ROOT/'.cache'/folder
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(path)
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle-name',default='serving-v1')
    parser.add_argument('--database',default='artifacts/phase6/demo.sqlite')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    import uvicorn
    from .api import create_app
    log_path=ROOT/'artifacts/phase7/runtime/requests.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    handler=RotatingFileHandler(log_path,maxBytes=1_000_000,backupCount=3,encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger=logging.getLogger('graphshield.requests')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate=False
    print(f'GraphShield AI: http://127.0.0.1:{args.port}',flush=True)
    uvicorn.run(create_app(args.bundle_name,args.database),host='127.0.0.1',port=args.port,workers=1,access_log=False)


if __name__=='__main__':
    main()
