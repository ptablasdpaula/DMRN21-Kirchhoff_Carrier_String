"""Differentiable CLAP scoring (laion/clap-htsat-unfused via HuggingFace transformers, offline).

Only the feature extractor of the HF pipeline is numpy; ``ClapModel`` itself
is an ordinary ``nn.Module``. ``ClapScorer`` reproduces the extractor in torch
(repeat-pad to 10 s, Hann/1024/480 power spectrogram, Slaney mel filters,
10·log10) so gradients reach the waveform, and exposes the Text2FX objectives.
"""

import os

import torch
import torch.nn.functional as F

MODEL_ID = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000
WINDOW = 480000  # 10 s
N_FFT, HOP = 1024, 480
MEL_FLOOR = 1e-10


class ClapScorer(torch.nn.Module):
    def __init__(self, device="cpu", model_id=MODEL_ID, dtype=torch.float32):
        super().__init__()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import ClapModel, ClapProcessor

        processor = ClapProcessor.from_pretrained(model_id)
        self.extractor, self.tokenizer = processor.feature_extractor, processor.tokenizer
        if (
            int(self.extractor.sampling_rate) != SAMPLE_RATE
            or int(self.extractor.hop_length) != HOP
            or int(self.extractor.fft_window_size) != N_FFT
            or int(self.extractor.nb_max_samples) != WINDOW
        ):
            raise RuntimeError("unexpected CLAP feature-extractor configuration")
        self.model = ClapModel.from_pretrained(model_id).to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.register_buffer(
            "mel_fb", torch.as_tensor(self.extractor.mel_filters_slaney, dtype=dtype, device=device)
        )  # (513, 64)
        self.register_buffer("window", torch.hann_window(N_FFT, periodic=True, dtype=dtype, device=device))
        self.device_, self.dtype_ = torch.device(device), dtype

    @staticmethod
    def normalise(audio, peak=0.5, eps=1e-12):
        """Peak-normalise each row (not detached: removes loudness as a lever)."""
        return audio / (audio.abs().amax(dim=-1, keepdim=True) + eps) * peak

    @staticmethod
    def tile(audio):
        """HF 'repeatpad' for clips shorter than 10 s: repeat floor(WINDOW/n) times, zero-pad the rest."""
        n = audio.shape[-1]
        if n > WINDOW:
            raise ValueError("clips longer than 10 s are not supported (CLAP would crop randomly)")
        repeats = WINDOW // n
        tiled = audio.repeat(1, repeats) if audio.ndim == 2 else audio.repeat(repeats)
        pad = WINDOW - tiled.shape[-1]
        return F.pad(tiled, (0, pad)) if pad else tiled

    def log_mel(self, x):
        """``(B, WINDOW)`` waveform -> ``(B, 1, frames, 64)`` dB log-mel, as the HF extractor computes it."""
        spectrum = torch.stft(
            x.to(self.dtype_),
            N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spectrum.real.square() + spectrum.imag.square()  # (B, 513, frames)
        mel = torch.einsum("bft,fm->bmt", power, self.mel_fb)  # (B, 64, frames)
        log_mel = 10 * torch.log10(mel.clamp_min(MEL_FLOOR))
        return log_mel.transpose(1, 2)[:, None]  # (B, 1, frames, 64)

    def embed_audio(self, audio, normalise=True):
        """``(B, samples)`` float audio (any dtype, <= 10 s) -> unit-norm ``(B, 512)`` CLAP audio embeddings."""
        x = audio.to(self.device_, self.dtype_)
        if normalise:
            x = self.normalise(x)
        features = self.log_mel(self.tile(x))
        out = self.model.get_audio_features(input_features=features, is_longer=None)
        return F.normalize(out.pooler_output, dim=-1)

    @torch.no_grad()
    def embed_text(self, prompts):
        tokens = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(self.device_)
        return F.normalize(self.model.get_text_features(**tokens).pooler_output, dim=-1)

    def reference_features(self, audio):
        """The HF numpy extractor on a ``(B, samples)`` array, for validation only."""
        import numpy as np

        rows = [np.asarray(row, dtype=np.float32) for row in audio.detach().cpu().numpy()]
        return self.extractor(rows, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_features"]


def cosine_loss(audio_embedding, text_embedding):
    """1 - cos between audio rows and a text embedding (broadcast)."""
    return 1 - (audio_embedding * text_embedding).sum(-1)


def directional_loss(audio_embedding, start_embedding, direction, tau=1e-3):
    """Text2FX directional objective: move the embedding along (target - contrast).

    ``direction`` is unit-norm; ``start_embedding`` is the detached embedding at
    iteration 0; ``tau`` keeps the loss finite when the displacement is zero.
    """
    delta = audio_embedding - start_embedding
    return 1 - (delta * direction).sum(-1) / (delta.norm(dim=-1) + tau)


def margin_loss(audio_embedding, start_embedding, direction):
    """Directional margin: 1 - (e - e0)·d_hat, i.e. one minus the displacement projected on the prompt direction.

    With d_hat = normalize(t_target - t_contrast) this maximises the similarity
    margin (s_plus - s_minus) relative to the start; it is exactly 1 at the
    start, so lowest-loss retention and plateau detection work from iteration 0.
    """
    return 1 - ((audio_embedding - start_embedding) * direction).sum(-1)


def barrier(logits, edge=3.0, weight=0.05):
    """Quadratic penalty outside |z| <= edge (about 95 % of each log range) to avoid sigmoid saturation."""
    return weight * F.relu(logits.abs() - edge).square().mean(-1)


def objective_backward(audio, per_item_loss, group=16):
    """Chunked loss/backward through CLAP, then one ``audio.backward`` so the string adjoint runs once.

    ``per_item_loss(chunk_audio, index_slice) -> (k,)`` must be differentiable in
    ``chunk_audio``. Returns the detached per-item losses ``(B,)``.
    """
    losses, grads = [], []
    for start in range(0, audio.shape[0], group):
        chunk = audio[start : start + group].detach().requires_grad_(True)
        value = per_item_loss(chunk, slice(start, start + group))
        (grad,) = torch.autograd.grad(value.sum(), chunk)
        losses.append(value.detach())
        grads.append(grad.to(audio.dtype))
    audio.backward(torch.cat(grads))
    return torch.cat(losses)
