"""Paired independent-vs-grouped CeL benchmarks for FP64 seeded CUDA FDTD."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.autograd.function import once_differentiable
from src.accelerated import evaluate
from src.optimization import criterion_for
from src.synthesis import Renderer
from benchmarks.exact_methods import BOX, GRID, coordinates_and_seeds, filter_audio, measure, metadata, varied_parameters
from experiments.loss_batch.grouped import grouped_loss_backward

torch.set_num_threads(1)

class Seeded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, steps, count):
        theta, seeds = coordinates_and_seeds(values)
        audio, jac = evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1,
                              'multimode', 'modal', True, seeds=seeds[:,:,:count])
        ctx.save_for_backward(jac)
        ctx.count = values.shape[1]
        return audio

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        jac, = ctx.saved_tensors
        g = torch.einsum('bt,btd->bd', grad.contiguous(), jac)
        out = grad.new_zeros((grad.shape[0], ctx.count))
        out[:,:g.shape[1]] = g
        return out, None, None

def relative(a, b):
    return (torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-30)).item()

def run(values, renderer, criterion, directions, group):
    p = values.detach().clone().requires_grad_()
    steps = round(renderer.duration * GRID.integration_rate)
    audio = filter_audio(Seeded.apply(p, steps, directions), renderer)
    if group is None:
        loss = torch.stack([criterion(row) for row in audio]).sum()
        loss.backward()
        loss = loss.detach()
    else:
        loss = grouped_loss_backward(audio, criterion, group)
    return loss, p.grad.detach()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batches', default='1,8,128')
    ap.add_argument('--durations', default='.25,4')
    ap.add_argument('--groups', default='4,8,16,32')
    ap.add_argument('--directions', default='2,6')
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    x = {'metadata': {**metadata(), 'status': 'running', 'method':'independent CeL versus groupwise CeL on forward sensitivities'}, 'checks': [], 'rows': []}
    a.output.parent.mkdir(exist_ok=True, parents=True)
    def save(): a.output.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
    save()
    for duration in map(float, a.durations.split(',')):
      target = Renderer(GRID, duration).synthesize(BOX.defaults()).detach().cuda()
      model = Renderer(GRID, duration, device='cuda')
      criterion = criterion_for(target, GRID.sample_rate)
      for batch in map(int, a.batches.split(',')):
       values = varied_parameters(batch).cuda()
       for directions in map(int, a.directions.split(',')):
        ref_loss, ref_grad = run(values, model, criterion, directions, None)
        torch.cuda.synchronize()
        row = {'batch':batch,'duration':duration,'directions':directions,'method':'independent'}
        row.update(measure(lambda: run(values, model, criterion, directions, None), values.device, a.repeats))
        x['rows'].append(row);save();print(json.dumps(row),flush=True)
        for group in map(int, a.groups.split(',')):
         try:
          loss, grad = run(values, model, criterion, directions, group)
          torch.testing.assert_close(loss, ref_loss, rtol=5e-10, atol=1e-12)
          torch.testing.assert_close(values*grad, values*ref_grad, rtol=1e-7, atol=1e-10)
          check={'batch':batch,'duration':duration,'directions':directions,'group':group,'status':'passed',
                 'relative_loss':relative(loss,ref_loss),'relative_scaled_gradient':relative(values*grad,values*ref_grad)}
          x['checks'].append(check);save()
          del loss, grad
          row={'batch':batch,'duration':duration,'directions':directions,'method':'grouped','group':group}
          row.update(measure(lambda: run(values, model, criterion, directions, group), values.device, a.repeats))
          x['rows'].append(row);save();print(json.dumps(row),flush=True)
         except Exception as e:
          x['checks'].append({'batch':batch,'duration':duration,'directions':directions,'group':group,'status':'failed','error':repr(e)});save();print('FAILED',x['checks'][-1],flush=True)
          if isinstance(e, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        del ref_loss, ref_grad
    x['metadata']['status']='complete';save()

if __name__ == '__main__': main()
