"""Measure the production raw-array forward/autograd API in fresh CPU processes.

Run: pixi run python benchmarks/native_autograd.py
Forward includes the external mean-square loss; backward is the saved-Jacobian
contraction. First-use build/load is separate. Output stride is one, preserving
all signal samples and eight tangents, including the two reference startup rows.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import autodiff_memory as b


def worker(duration):
    b.load_torch()
    torch = b.torch
    from kc_string_forward import ForwardString
    # Match the other benchmarks' warmed Python/PyTorch baseline.
    warm = torch.ones(8, dtype=torch.float64, requires_grad=True)
    b.objective(b.simulate(warm*b.defaults(),32,128,44100,'checkpoint',16)).backward()
    del warm
    model = ForwardString()
    theta = torch.ones(8, dtype=torch.float64, requires_grad=True)
    started = time.perf_counter()
    b.objective(model(theta,256/44100)).backward()
    first_use = time.perf_counter()-started
    theta.grad = None
    baseline = b.peak_mib()
    started = time.perf_counter()
    audio = model(theta,duration)
    loss = b.objective(audio)
    forward = time.perf_counter()-started
    forward_peak = b.peak_mib()
    started = time.perf_counter()
    loss.backward()
    backward = time.perf_counter()-started
    assert torch.isfinite(theta.grad).all() and torch.isfinite(loss)
    print(json.dumps(dict(method='native_autograd',duration_seconds=duration,
        forward_seconds=forward,backward_seconds=backward,total_seconds=forward+backward,
        first_use_seconds=first_use,baseline_peak_mib=baseline,forward_peak_mib=forward_peak,
        total_peak_mib=b.peak_mib(),additional_peak_mib=b.peak_mib()-baseline,
        loss=loss.item(),gradients=theta.grad.tolist(),output_samples=audio.numel(),
        saved_tangent_mib=audio.numel()*8*8/2**20,
        threads=1,dtype='float64',device='cpu',sample_rate=44100,intervals=128,output_stride=1)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--duration',type=float,default=.06)
    parser.add_argument('--output',default='.notebook-build/native_autograd.json')
    args=parser.parse_args()
    if args.worker:
        worker(args.duration)
    else:
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        records=json.loads(path.read_text()) if path.exists() else []
        for duration in [.06,.25,1.,4.]:
            for repetition in range(3):
                if any(r['duration_seconds']==duration and r['repetition']==repetition for r in records):
                    continue
                result=subprocess.run([sys.executable,__file__,'--worker','--duration',str(duration)],capture_output=True,text=True,timeout=120)
                if result.returncode:
                    raise RuntimeError(result.stderr+result.stdout)
                record=json.loads(result.stdout.strip().splitlines()[-1])
                record['repetition']=repetition
                records.append(record)
                path.write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(record),flush=True)
