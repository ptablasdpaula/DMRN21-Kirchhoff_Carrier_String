"""Check experimental grouped CeL matches summed independent losses."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from cels import STFTCumulativeEnergyLoss
from experiments.loss_batch.grouped import grouped_loss_backward
from src.optimization import criterion_for

torch.manual_seed(73);torch.set_num_threads(1)
target=torch.randn(2048,dtype=torch.float64)
criterion=STFTCumulativeEnergyLoss(sample_rate=44100).bind(target)
values=target.repeat(7,1)+.01*torch.randn(7,2048,dtype=torch.float64)
for group in (1,2,3,7,8):
    ref=values.clone().requires_grad_();x=values.clone().requires_grad_()
    loss=torch.stack([criterion(row) for row in ref]).sum();refgrad,=torch.autograd.grad(loss,ref)
    got=grouped_loss_backward(x,criterion,group)
    torch.testing.assert_close(got,loss.detach(),rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(x.grad,refgrad,rtol=1e-8,atol=1e-11)
batched=STFTCumulativeEnergyLoss(sample_rate=44100).bind(values)
x=values.clone().requires_grad_()
try:grouped_loss_backward(x,batched,4)
except ValueError as error:
    assert 'single common bound target' in str(error)
else:raise AssertionError('Per-item bound targets must be rejected')
# The waveform MAE shares the bound/mean-reduction contract: the grouped value is
# the sum of per-item means and its gradient is autograd's for that sum.
mae=criterion_for(target,44100,loss='mae')
torch.testing.assert_close(mae(values),(values-target).abs().mean(dim=-1).mean(),rtol=1e-12,atol=1e-14)
for group in (1,2,3,7,8):
    ref=values.clone().requires_grad_();x=values.clone().requires_grad_()
    loss=torch.stack([mae(row) for row in ref]).sum();refgrad,=torch.autograd.grad(loss,ref)
    torch.testing.assert_close(loss.detach(),(values-target).abs().mean(dim=-1).sum(),rtol=1e-12,atol=1e-14)
    got=grouped_loss_backward(x,mae,group)
    torch.testing.assert_close(got,loss.detach(),rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(x.grad,refgrad,rtol=1e-12,atol=1e-14)
print('PASS grouped CeL and waveform MAE common target and per-item target rejection')
