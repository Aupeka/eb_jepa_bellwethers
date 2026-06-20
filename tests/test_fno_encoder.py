"""Tests for the FNO encoder (``eb_jepa.architectures.FNOEncoder``).

Covers:
1. Shape preservation for 4D ``[B, C, H, W]`` and 5D ``[B, C, T, H, W]`` inputs.
2. The ``out_d`` attribute matches the configured latent dim.
3. Safety under ``torch.amp.autocast(bfloat16)`` (the FFT must run in fp32).
4. Integration: a JEPA built with the FNO encoder runs one ``unroll`` step.
"""

import torch

from eb_jepa.architectures import (
    FNOEncoder,
    Projector,
    ResUNet,
    SpectralConv2d,
    StateOnlyPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VCLoss


def test_spectral_conv_preserves_resolution():
    """SpectralConv2d maps [B, C, H, W] -> [B, C_out, H, W] at the same resolution."""
    layer = SpectralConv2d(in_channels=4, out_channels=6, modes1=8, modes2=8)
    x = torch.randn(2, 4, 32, 32)
    out = layer(x)
    assert out.shape == (2, 6, 32, 32)
    assert out.dtype == torch.float32


def test_fno_encoder_4d_shape():
    """4D frame input [B, in_d, H, W] -> [B, out_d, H, W] (resolution preserved)."""
    enc = FNOEncoder(in_d=2, h_d=32, out_d=16, modes=16, n_layers=4)
    x = torch.randn(2, 2, 128, 128)
    out = enc(x)
    assert out.shape == (2, 16, 128, 128)


def test_fno_encoder_5d_shape():
    """5D clip [B, in_d, T, H, W] -> [B, out_d, T, H, W] via TemporalBatchMixin."""
    enc = FNOEncoder(in_d=2, h_d=32, out_d=16, modes=16, n_layers=4)
    x = torch.randn(2, 2, 5, 128, 128)
    out = enc(x)
    assert out.shape == (2, 16, 5, 128, 128)


def test_fno_encoder_exposes_out_d():
    """The encoder exposes out_d for downstream wiring."""
    enc = FNOEncoder(in_d=2, h_d=8, out_d=16, modes=4, n_layers=2)
    assert enc.out_d == 16


def test_fno_encoder_autocast_bf16_safe():
    """Encoder runs under bfloat16 autocast (FFT is guarded to fp32)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = FNOEncoder(in_d=2, h_d=16, out_d=8, modes=8, n_layers=2).to(device)
    x = torch.randn(2, 2, 64, 64, device=device)
    with torch.amp.autocast(device, dtype=torch.bfloat16):
        out = enc(x)
    assert out.shape == (2, 8, 64, 64)


def test_resunet_decoder_latent_to_field_shape():
    """ResUNet decoder maps latent [B, D, T, H, W] -> field [B, 2, T, H, W]."""
    dec = ResUNet(16, 32, 2)
    z4 = torch.randn(2, 16, 64, 64)
    assert dec(z4).shape == (2, 2, 64, 64)
    z5 = torch.randn(2, 16, 5, 64, 64)
    assert dec(z5).shape == (2, 2, 5, 64, 64)


def test_spectral_decoder_latent_to_field_shape():
    """Spectral decoder (FNOEncoder with out_d=2) maps latent -> 2-channel field."""
    dec = FNOEncoder(in_d=16, h_d=32, out_d=2, modes=8, n_layers=2)
    z4 = torch.randn(2, 16, 64, 64)
    assert dec(z4).shape == (2, 2, 64, 64)
    z5 = torch.randn(2, 16, 5, 64, 64)
    assert dec(z5).shape == (2, 2, 5, 64, 64)


def test_fno_encoder_in_jepa_unroll():
    """A JEPA built with the FNO encoder runs one parallel unroll step with loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dstc, hpre = 16, 16
    encoder = FNOEncoder(in_d=2, h_d=16, out_d=dstc, modes=8, n_layers=2)
    predictor = StateOnlyPredictor(ResUNet(2 * dstc, hpre, dstc), context_length=2)
    projector = Projector(f"{dstc}-{dstc * 4}-{dstc * 4}")
    regularizer = VCLoss(std_coeff=10.0, cov_coeff=100.0, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    x = torch.randn(2, 2, 8, 64, 64, device=device)
    _, losses = jepa.unroll(
        x, actions=None, nsteps=4, unroll_mode="parallel", compute_loss=True
    )
    loss, rloss, _, _, pl = losses
    assert loss.shape == torch.Size([])
    assert rloss.shape == torch.Size([])
    assert pl.shape == torch.Size([])
