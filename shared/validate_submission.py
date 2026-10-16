from __future__ import annotations
import argparse, json, math, tempfile, zipfile
from pathlib import Path
import torch
from shared.contract import CONTRACT_VERSION, OBSERVATION_SIZE, ACTION_SIZE

REQUIRED = {"model.pt", "metadata.json"}
REQUIRED_META = {
    "team_id", "team_name", "members", "model_format", "contract_version",
    "observation_size", "action_size", "training_algorithm", "notes"
}

def validate_zip(path: Path, quiet=False):
    errors=[]
    if not path.exists() or path.suffix.lower() != '.zip':
        return False, [f"Not a ZIP file: {path}"], None
    try:
        with zipfile.ZipFile(path) as z:
            names={n for n in z.namelist() if not n.endswith('/')}
            if names != REQUIRED:
                errors.append(f"ZIP root must contain exactly {sorted(REQUIRED)}; found {sorted(names)}")
            if 'metadata.json' not in names or 'model.pt' not in names:
                return False, errors, None
            meta=json.loads(z.read('metadata.json').decode('utf-8'))
            missing=REQUIRED_META-set(meta)
            if missing: errors.append(f"metadata.json missing keys: {sorted(missing)}")
            if meta.get('model_format')!='torchscript': errors.append("model_format must be 'torchscript'")
            if meta.get('contract_version')!=CONTRACT_VERSION: errors.append(f"contract_version must be {CONTRACT_VERSION}")
            if meta.get('observation_size')!=OBSERVATION_SIZE: errors.append(f"observation_size must be {OBSERVATION_SIZE}")
            if meta.get('action_size')!=ACTION_SIZE: errors.append(f"action_size must be {ACTION_SIZE}")
            if not isinstance(meta.get('members'), list): errors.append('members must be a JSON list')
            if not str(meta.get('team_id','')).strip(): errors.append('team_id cannot be empty')
            with tempfile.TemporaryDirectory() as td:
                mp=Path(td)/'model.pt'; mp.write_bytes(z.read('model.pt'))
                try:
                    model=torch.jit.load(str(mp),map_location='cpu').eval()
                    x=torch.zeros((1,OBSERVATION_SIZE),dtype=torch.float32)
                    with torch.inference_mode(): y=model(x)
                    if not isinstance(y,torch.Tensor): errors.append('model output must be a torch.Tensor')
                    elif tuple(y.shape)!=(1,ACTION_SIZE): errors.append(f'model output shape must be (1,{ACTION_SIZE}); found {tuple(y.shape)}')
                    elif not torch.isfinite(y).all(): errors.append('model output contains NaN/Inf')
                except Exception as e:
                    errors.append(f'Cannot load/run model.pt as TorchScript: {type(e).__name__}: {e}')
    except Exception as e:
        errors.append(f'Invalid ZIP: {type(e).__name__}: {e}')
        meta=None
    ok=not errors
    if not quiet:
        print(('VALID' if ok else 'INVALID') + f': {path}')
        for e in errors: print(' -',e)
    return ok, errors, meta

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('zipfile',type=Path); a=p.parse_args()
    ok,_,_=validate_zip(a.zipfile)
    raise SystemExit(0 if ok else 2)
