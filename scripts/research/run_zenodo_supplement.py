"""Run the registered experiments sequentially and retain exact child handles."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys

from zenodo_supplement_common import read, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--phase',choices=['classification'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-queue',action='store_true')
    args=parser.parse_args()
    registry=read(args.registry)
    status_path=args.output/'queue_status.json'
    if status_path.exists() and not args.resume_queue:
        raise ValueError('A queue status already exists; inspect its actual process before another launch')
    status={'pid':os.getpid(),'phase':args.phase,'completed':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.resume_queue:
        previous=read(status_path)
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Previous queue process still exists; do not duplicate it')
        status['previous_status']=previous
    jobs=[]
    for backbone in ('resnet18','vit_b16'):
        jobs.append((['--backbone',backbone,'--class-weighting','balanced','--views','global','--check-only'],
                     args.output/'checks'/backbone))
    for run in registry['classification']:
        jobs.append((['--backbone',run['backbone'],'--class-weighting',run['class_weighting'],'--views',run['views']],
                     args.output/'classification'/run['name']/'seed42/attempt1'))
    for options,folder in jobs:
        if folder.exists():
            completed=folder/('checks.json' if '--check-only' in options else 'done.json')
            if completed.exists():
                status['completed'].append(str(folder));continue
            suffix=2
            original=folder
            while folder.exists():
                folder=original.with_name(original.name+'-retry'+str(suffix));suffix+=1
        command=[sys.executable,'-u','scripts/research/train_zenodo_classifier.py','--config',
                 'configs/research/zenodo_supplement/classifier.json',*options,'--output',str(folder)]
        folder.parent.mkdir(parents=True,exist_ok=True)
        log=folder.parent/(folder.name+'.log')
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            status.update(state='running',child_pid=child.pid,current_run=str(folder),command=command,log=str(log))
            write(status_path,status)
            print(f'Running PID {child.pid}: {folder}',flush=True)
            code=child.wait()
        if code:
            status.update(state='failed',returncode=code)
            write(status_path,status)
            raise RuntimeError(f'Experiment failed: {folder}; see {log}')
        status['completed'].append(str(folder))
        write(status_path,status)
    status.update(state='completed',child_pid=None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(status_path,status)
    print('Classification queue completed',flush=True)


if __name__=='__main__':
    main()
