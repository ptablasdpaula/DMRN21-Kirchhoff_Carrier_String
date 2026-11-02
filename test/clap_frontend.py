"""CPU check: the torch CLAP front end matches the HuggingFace extractor and keeps gradients (skips without the model)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
try:
    import transformers  # noqa: F401
    from experiments.applications.semantic.clap import ClapScorer, cosine_loss, directional_loss, barrier, SAMPLE_RATE
    scorer = ClapScorer('cpu')
except Exception as error:   # the model or the package may be absent on this machine
    print(f'SKIP clap_frontend: {error!r}')
    sys.exit(0)

torch.manual_seed(0)
clips = {'2.5s': torch.randn(2, SAMPLE_RATE*5//2, dtype=torch.float64)*0.1,
         '1.0s': torch.randn(2, SAMPLE_RATE, dtype=torch.float64)*0.1}      # 1 s exercises the zero padding
for name, audio in clips.items():
    ours = scorer.log_mel(scorer.tile(audio.float()))
    theirs = scorer.reference_features(audio).to(ours.dtype)
    assert ours.shape == theirs.shape == (2, 1, 1001, 64), (ours.shape, theirs.shape)
    clamp = lambda t: t.clamp_min(-90.0)
    torch.testing.assert_close(clamp(ours), clamp(theirs), rtol=1e-4, atol=5e-3)
    with torch.no_grad():
        ref = torch.nn.functional.normalize(scorer.model.get_audio_features(input_features=theirs, is_longer=None).pooler_output, dim=-1)
        emb = scorer.embed_audio(audio, normalise=False)
    cos = (emb*ref).sum(-1)
    assert bool((cos > 0.9999).all()), cos
    print(f'{name}: log-mel max |diff| {float((ours-theirs).abs().max()):.2e} dB, embedding cosine {cos.min():.6f}')
audio = clips['2.5s'][:1].clone().requires_grad_(True)
emb = scorer.embed_audio(audio)
assert emb.shape == (1, 512) and abs(float(emb.norm())-1) < 1e-5
emb[0, 0].backward()
assert audio.grad is not None and bool(torch.isfinite(audio.grad).all()) and float(audio.grad.abs().max()) > 0
with torch.no_grad():
    loud = scorer.embed_audio(clips['2.5s']*10)
    quiet = scorer.embed_audio(clips['2.5s'])
torch.testing.assert_close(loud, quiet, rtol=0, atol=1e-5)                       # peak normalisation removes level
texts = scorer.embed_text(['a metallic plucked steel string', 'a dull, muted nylon string'])
assert texts.shape == (2, 512) and float((texts[0]*texts[1]).sum()) < 0.999
direction = torch.nn.functional.normalize(texts[0]-texts[1], dim=-1)
start = quiet.detach()
loss0 = directional_loss(quiet, start, direction)
assert bool(torch.isfinite(loss0).all()) and bool((cosine_loss(quiet, texts[0]) > 0).all())
assert float(barrier(torch.zeros(3, 9, dtype=torch.float64))) == 0 and float(barrier(torch.full((1, 9), 4.0))) > 0
print('clap_frontend: all checks passed')
