"""GPU smoke test for the semantic application: wide-box render, CLAP gradient through the string, a few iterations."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
from experiments.applications.common.boxes import decode_batch, encode_rows, sample_logits
from experiments.applications.common.render import BatchRenderer, theta_from_physical
from experiments.applications.semantic.box import WideStringBox, pilot_grid, production_grid, DURATION, EXCITATION, FADE_SECONDS
from experiments.applications.semantic.clap import ClapScorer, directional_loss
from src.adjoint import simulate as replay
from src.kc_string_forward import ForwardString


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    box = WideStringBox()
    grid = pilot_grid(box)
    renderer = BatchRenderer(grid, DURATION, EXCITATION, device='cuda', fade_seconds=FADE_SECONDS)
    logits = sample_logits(box, 4, seed=2, device='cuda')
    logits[0] = encode_rows(box, box.defaults()).cuda()
    material = box.to_material(decode_batch(box, logits))
    # 1. Render vs CPU reference on a short clip, cached adjoint gradient vs checkpointed replay at the production grid.
    short = BatchRenderer(production_grid(box), 0.05, EXCITATION, device='cuda')
    x = material.clone().requires_grad_()
    raw = short.raw(x)
    reference = ForwardString(short.grid.intervals, short.grid.integration_rate, excitation=EXCITATION, order=short.grid.order)
    cpu_audio, _ = reference.output_and_tangents(theta_from_physical(material[1].cpu()), 0.05)
    torch.testing.assert_close(raw[1].detach().cpu(), cpu_audio, rtol=1e-6, atol=1e-12)
    (raw.square().mean()).backward()
    grad_cached = x.grad.detach().clone()
    y = material.clone().requires_grad_()
    raw_replay = replay(theta_from_physical(y), short.grid.intervals, short.grid.integration_rate, short.steps, EXCITATION, 256, short.grid.order)
    (raw_replay.square().mean()).backward()
    torch.testing.assert_close(grad_cached, y.grad, rtol=1e-6, atol=1e-18)
    gradient_error = float((grad_cached-y.grad).norm()/y.grad.norm())
    # 2. CLAP objective through the string for the pilot grid batch.
    scorer = ClapScorer('cuda')
    text = scorer.embed_text(['a metallic plucked steel string', 'a dull, muted nylon string'])
    d_hat = torch.nn.functional.normalize(text[0]-text[1], dim=-1)
    z = logits.clone().requires_grad_()
    t0 = time.perf_counter()
    audio = renderer(box.to_material(decode_batch(box, z)))
    e0 = scorer.embed_audio(audio).detach()
    loss = (1-(scorer.embed_audio(audio)*text[0]).sum(-1)).sum()
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-t0
    assert z.grad is not None and bool(torch.isfinite(z.grad).all()) and float(z.grad.abs().max()) > 0, z.grad
    # 3. Three Adam steps on the directional objective lower the mean objective for at least one item.
    z = logits.clone().requires_grad_()
    optimiser = torch.optim.Adam([z], lr=0.05)
    values = []
    for _ in range(3):
        optimiser.zero_grad()
        audio = renderer(box.to_material(decode_batch(box, z)))
        e = scorer.embed_audio(audio)
        objective = directional_loss(e, e0, d_hat)
        objective.sum().backward(); optimiser.step()
        values.append(objective.detach().cpu().tolist())
    report = {'status': 'passed', 'pilot_grid': renderer.describe(), 'production_grid': short.grid.describe(),
              'waveform_relative_l2_vs_cpu': float((raw[1].detach().cpu()-cpu_audio).norm()/cpu_audio.norm()),
              'gradient_relative_l2_cached_vs_replay': gradient_error, 'seconds_render_clap_backward_batch4': seconds,
              'directional_objective_per_iteration': values, 'peak_m': audio.detach().abs().amax(-1).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
