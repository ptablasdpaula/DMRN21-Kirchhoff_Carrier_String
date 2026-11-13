"""Compare the JAX default filtering/CeL loss and waveform gradient to torch."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import torch
from cels import STFTCumulativeEnergyLoss
from experiments.jax_loss_benchmark import anti_alias, cel, jax_loss, mae, prepare_mae_target, prepare_target
from src.optimization import criterion_for


def run(device, output):
    if device == 'cuda' and (not torch.cuda.is_available() or jax.devices()[0].platform != 'gpu'):
        raise RuntimeError('GPU requested, but JAX and PyTorch must both see a GPU')
    torch_device = torch.device(device)
    generator = np.random.default_rng(3927)
    oversampling = 3
    half_width = 16 * oversampling
    index = torch.arange(-half_width, half_width + 1, dtype=torch.float64)
    cutoff = .45 / oversampling
    kernel = 2 * cutoff * torch.sinc(2 * cutoff * index) * torch.hann_window(
        index.numel(), periodic=False, dtype=torch.float64)
    kernel = (kernel / kernel.sum()).to(torch_device)
    jkernel = jnp.asarray(kernel.cpu().numpy())
    checks = []
    for length, batch, near in [(11025, 1, False), (11025, 3, False),
                                (11025, 1, True), (176400, 1, False)]:
        # Independent nonzero candidates and targets avoid degenerate zero loss.
        wave = generator.normal(0, .01, (batch, length * oversampling + 2))
        target = generator.normal(0, .005, (batch, length))
        tx = torch.tensor(wave, dtype=torch.float64, device=torch_device, requires_grad=True)
        padded = torch.nn.functional.pad(tx[:, None, :], (half_width, half_width), mode='replicate')
        filtered = torch.nn.functional.conv1d(padded, kernel[None, None, :], stride=oversampling)[:, 0, :length]
        if near:
            target = filtered.detach().cpu().numpy() + generator.normal(0, .00005, (batch, length))
        ty = torch.tensor(target, dtype=torch.float64, device=torch_device)
        # Compute independent per-item losses, then average, exactly as cel does.
        torch_loss = torch.stack([STFTCumulativeEnergyLoss(sample_rate=44100).to(torch_device)
                                  .bind(ty[b])(filtered[b]) for b in range(batch)]).mean()
        (torch_gradient,) = torch.autograd.grad(torch_loss, tx)

        jx, jy = jnp.asarray(wave), jnp.asarray(target)
        prepared = prepare_target(jy)
        jfilter = jax.jit(lambda z: anti_alias(z, jkernel, length, oversampling))(jx)
        jloss, jgradient = jax.jit(jax.value_and_grad(
            lambda z: jax_loss(z, prepared, jkernel, oversampling)))(jx)
        jloss.block_until_ready()
        jgradient.block_until_ready()
        tf = filtered.detach().cpu().numpy()
        tg = torch_gradient.cpu().numpy()
        jf, jg = np.asarray(jfilter), np.asarray(jgradient)
        jl, tl = float(jloss), float(torch_loss.detach().cpu())
        rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
        np.testing.assert_allclose(jf, tf, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(jl, tl, rtol=2e-8, atol=1e-11)
        np.testing.assert_allclose(jg, tg, rtol=2e-6, atol=1e-11)
        checks.append(dict(length=length, batch=batch, near_target=near, filter_relative_l2=rel(jf, tf),
                           loss_relative=abs(jl-tl)/abs(tl), gradient_relative_l2=rel(jg,tg)))
        print(json.dumps(checks[-1]), flush=True)
    # Zero residual must have finite zero gradient, as torch's norm does.
    target = jnp.asarray(generator.normal(0, .005, (2048,)))
    prepared = prepare_target(target)
    gradient = jax.grad(lambda z: cel(z, prepared))(target)
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.max(np.abs(np.asarray(gradient))) == 0
    # The waveform MAE on the same random audio and target in both frameworks:
    # the mean over items of the per-item mean |audio - target|, and its gradient.
    audio = generator.normal(0, .01, (3, 4096))
    target = generator.normal(0, .005, (4096,))
    ta = torch.tensor(audio, dtype=torch.float64, device=torch_device, requires_grad=True)
    torch_mae = criterion_for(torch.tensor(target, dtype=torch.float64, device=torch_device), 44100, loss='mae')(ta)
    (torch_mae_gradient,) = torch.autograd.grad(torch_mae, ta)
    prepared = prepare_mae_target(jnp.asarray(target))
    jax_mae, jax_mae_gradient = jax.jit(jax.value_and_grad(lambda z: mae(z, prepared)))(jnp.asarray(audio))
    tl, jl = float(torch_mae.detach().cpu()), float(jax_mae)
    np.testing.assert_allclose(jl, tl, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax_mae_gradient), torch_mae_gradient.cpu().numpy(), rtol=1e-12, atol=1e-12)
    checks.append(dict(loss='mae', length=4096, batch=3, loss_relative=abs(jl-tl)/abs(tl),
                       gradient_relative_l2=rel(np.asarray(jax_mae_gradient), torch_mae_gradient.cpu().numpy())))
    print(json.dumps(checks[-1]), flush=True)
    metadata = dict(device=device, torch=torch.__version__, jax=jax.__version__,
                    jax_devices=[str(d) for d in jax.devices()], checks=checks,
                    zero_loss_gradient='passed', status='passed')
    Path(output).write_text(json.dumps(metadata, indent=2) + '\n')
    print('ALL DEFAULT JAX CeL AND MAE CHECKS PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.device, args.output)
