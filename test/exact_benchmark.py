"""Verify research harness represents independent loss and VJP for each string."""
import argparse
import json
import os
import time
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.exact_methods import BOX, GRID, audio_for, varied_parameters, digest
from src.optimization import criterion_for
from src.synthesis import Renderer
from src.accelerated import kernel
from src.kc_string_forward import _native_kernel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--method')
    parser.add_argument('--duration', type=float, default=.03)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    duration=args.duration
    batch=args.batch
    source_hash=digest()
    checks=[]
    build={}
    started=time.perf_counter();_native_kernel();build['native_load_or_build_seconds']=time.perf_counter()-started
    started=time.perf_counter();kernel(args.device.startswith('cuda'));build['fused_load_or_build_seconds']=time.perf_counter()-started
    if args.adjoint:
        from src.adjoint import kernel as adjoint_kernel
        started=time.perf_counter();adjoint_kernel();build['adjoint_load_or_build_seconds']=time.perf_counter()-started
    print('Extension load/build timings:',build,flush=True)
    values=varied_parameters(batch)
    cpu=Renderer(GRID,duration)
    target=cpu.synthesize(BOX.defaults())
    criterion=criterion_for(target,GRID.sample_rate)
    x=values.clone().requires_grad_()
    reference=torch.stack([cpu(row) for row in x])
    losses=torch.stack([criterion(row) for row in reference])
    losses.sum().backward()
    ref_audio=reference.detach()
    ref_grad=x.grad.detach()
    methods=['cpu-native','cpu-modal']
    if args.device.startswith('cuda'):
        methods+=['cuda-modal','cuda-recompute','cuda-register','cuda-seeded']
        if args.adjoint: methods+=['cuda-adjoint']
    if args.method:
        selected=args.method.split(',')
        if any(method not in methods for method in selected): raise ValueError(f'Invalid method for device: {args.method}')
        methods=selected
    for method in methods:
        device='cuda' if method.startswith('cuda') else 'cpu'
        renderer=Renderer(GRID,duration,device=device)
        p=values.to(device).clone().requires_grad_()
        with torch.no_grad():
            primal=audio_for(p,renderer,method,tangents=False)
        torch.testing.assert_close(primal.cpu(),ref_audio,rtol=1e-6,atol=1e-10)
        if method!='cuda-register':
            audio=audio_for(p,renderer,method,tangents=True)
            loss=criterion_for(target.to(device),GRID.sample_rate)
            actual_losses=torch.stack([loss(row) for row in audio])
            actual_losses.sum().backward()
            torch.testing.assert_close(audio.detach().cpu(),ref_audio,rtol=1e-6,atol=1e-10)
            torch.testing.assert_close(actual_losses.detach().cpu(),losses.detach(),rtol=5e-4,atol=1e-10)
            # Relative-coordinate scaling makes small absolute physical values comparable.
            torch.testing.assert_close((p.grad*values.to(device)).cpu(),ref_grad*values,rtol=5e-4,atol=1e-8)
            checks.append(dict(method=method,waveform='passed',independent_losses='passed',gradients='passed',waveform_relative_l2=((audio.detach().cpu()-ref_audio).norm()/ref_audio.norm()).item()))
            print(method,'waveform/independent losses/gradients PASS',flush=True)
        else:
            checks.append(dict(method=method,waveform='passed',waveform_relative_l2=((primal.cpu()-ref_audio).norm()/ref_audio.norm()).item()))
            print(method,'primal PASS',flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(source_sha256=source_hash,source_changed_during_run=digest()!=source_hash,duration=duration,batch=batch,status='passed',checks=checks,extension_load_or_build_seconds=build,max_build_jobs=os.environ.get('MAX_JOBS')),indent=2)+'\n')
    print('EXACT BENCHMARK CHECKS PASS',flush=True)

if __name__=='__main__': main()
