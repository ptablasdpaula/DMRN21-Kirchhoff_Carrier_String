"""Single-string 4 s cached-adjoint benchmark at a guitar-range grid.

Forward/backward timing includes raw-output mean-square loss and CUDA sync.
Reports PyTorch allocated tensor peak, not full NVML process memory.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.scalar_cache.reconstruct import simulate
from src.kc_string_forward import nominal_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA worker.')
    rate, n, steps = 44100*55, 176, 44100*55*4
    length, tension, rho, f0, b = .1625, 70.3, 8050., 1318.4, .001984
    mu = tension/(2*length*f0)**2
    EI = b*tension*length**2/math.pi**2
    radius = math.sqrt(mu/(math.pi*rho))
    young = 4*EI/(math.pi*radius**4)
    params = torch.tensor([rho,tension,radius,young,length,.002,1.,.001],dtype=torch.float64,device='cuda')
    theta = params/nominal_parameters('cuda')
    # JIT compilation and context startup do not enter the measured interval.
    for count in (1,3):
        warm = theta.detach().clone().requires_grad_()
        simulate(warm,n,rate,512,'raised_cosine',256,count).square().mean().backward()
    torch.cuda.synchronize()
    rows = []
    for count in (1,3):
        x=theta.detach().clone().requires_grad_()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        raw=simulate(x,n,rate,steps,'raised_cosine',256,count)
        loss=raw.square().mean()/(.002**2)
        torch.cuda.synchronize()
        forward=time.perf_counter()-start
        start=time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward=time.perf_counter()-start
        if not bool(torch.isfinite(raw).all() and torch.isfinite(x.grad).all()):
            raise RuntimeError('Non-finite waveform or gradients.')
        rows.append({'cached_scalars':count,'forward_seconds':forward,
                     'backward_seconds':backward,'combined_seconds':forward+backward,
                     'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                     'loss':float(loss.detach()),'gradients':x.grad.detach().cpu().tolist()})
        print(json.dumps(rows[-1]),flush=True)
        del raw,loss,x
    result={'gpu':torch.cuda.get_device_name(), 'duration_s':4.0,'batch':1,
            'intervals':n,'output_rate_hz':44100,'integration_rate_hz':rate,
            'excitation':'raised_cosine','case':'conservative independent guitar parameter box corner',
            'output':'raw internal-rate midpoint; mean-square objective, no filtering',
            'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
