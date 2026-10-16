from __future__ import annotations
import argparse, json, re, shutil, tempfile, zipfile
from pathlib import Path
import torch
from shared.contract import metadata_template, OBSERVATION_SIZE, ACTION_SIZE
from shared.validate_submission import validate_zip

def safe_team_id(x: str) -> str:
    x=re.sub(r'[^A-Za-z0-9_-]+','_',x.strip().upper())
    if not x: raise ValueError('team_id cannot be empty')
    return x

def main():
    p=argparse.ArgumentParser(description='Create the exact competition ZIP.')
    p.add_argument('--team-id',required=True,help='Example: TEAM_05')
    p.add_argument('--team-name',default='')
    p.add_argument('--members',default='',help='Comma-separated names')
    p.add_argument('--algorithm',required=True,help='Example: DQN, PPO, MAPPO')
    p.add_argument('--model',type=Path,default=Path('outputs/model.pt'))
    p.add_argument('--notes',default='')
    a=p.parse_args()
    team=safe_team_id(a.team_id)
    if not a.model.exists(): raise SystemExit(f'Model not found: {a.model}')
    # Load once before packaging so an invalid file is never exported.
    m=torch.jit.load(str(a.model),map_location='cpu').eval()
    with torch.inference_mode():
        y=m(torch.zeros((1,OBSERVATION_SIZE),dtype=torch.float32))
    if tuple(y.shape)!=(1,ACTION_SIZE):
        raise SystemExit(f'Model output must be (1,{ACTION_SIZE}), got {tuple(y.shape)}')
    meta=metadata_template(team,a.team_name,[x.strip() for x in a.members.split(',') if x.strip()])
    meta['training_algorithm']=a.algorithm.strip()
    meta['notes']=a.notes[:500]
    outdir=Path('submission'); outdir.mkdir(exist_ok=True)
    out=outdir/f'{team}.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(a.model,arcname='model.pt')
        z.writestr('metadata.json',json.dumps(meta,indent=2,ensure_ascii=False))
    ok,errors,_=validate_zip(out,quiet=True)
    if not ok:
        out.unlink(missing_ok=True)
        raise SystemExit('Export failed validation: '+'; '.join(errors))
    print(f'Created and validated: {out.resolve()}')
    print('ZIP contents: model.pt, metadata.json')

if __name__=='__main__': main()
