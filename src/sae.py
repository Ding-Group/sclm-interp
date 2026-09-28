"""
Sparse Autoencoders with variance-normalized losses.

This module supersedes sae_legacy.py. The architectures are unchanged; only the
loss scaling differs, following the reference implementation of
'Scaling and Evaluating Sparse Autoencoders' (Gao et al. 2024,
github.com/openai/sparse_autoencoder):

    -   Reconstruction loss is MSE scaled by 1 / Var(x), where Var(x) is the
        per-element variance of a data sample captured at init (their mse_scale).
        This makes training losses comparable across base models and layers
        regardless of raw activation magnitude (loss ~= fraction of variance
        unexplained).
    -   The TopK auxiliary dead-feature loss is normalized by the per-batch
        variance of its target residual (their normalized_mse), keeping the
        recon/aux balance scale-free throughout training.
    -   The vanilla L1 penalty is normalized by the input norm per sample
        (their normalized_L1_loss), so l1_coeff is on a scale-free footing.

The scale is estimated from init_data (the same batch used for b_dec init).
If init_data is None, mse_scale stays 1.0 and losses reduce to raw MSE, i.e.
legacy behavior. mse_scale is a non-persistent buffer: state_dicts remain
key-compatible with sae_legacy.py checkpoints in both directions.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


# Fixed TopK dead-feature threshold used by the original training runs.  Keep
# this independent of k and d_hidden so changing the sparsity or expansion does
# not silently change which features receive the auxiliary revival loss.
_DEAD_RATE_THRESHOLD = 1e-4


class SAE(nn.Module):
    """
    Base class for Sparse Autoencoders.
    Handles shared weight initialization, decode, normalize, and forward.
    Subclasses must implement encode() and loss_function().
    """

    def __init__(self, d_model: int, expansion: int, weight_tying: bool = False, init_data: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion
        self.weight_tying = weight_tying

        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        # Decoder is stored as (d_hidden, d_model).
        # Decode uses x @ w_dec + b_dec.
        self.w_dec = nn.Parameter(torch.empty(self.d_hidden, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        # Encoder weight shape for F.linear(x, w_enc, b_enc): (d_hidden, d_model)
        if not weight_tying:
            self.w_enc = nn.Parameter(torch.empty(self.d_hidden, self.d_model))

        # 1 / Var(x) of the training data, estimated from init_data. Scales the
        # reconstruction MSE so the loss reads as fraction of variance
        # unexplained. Non-persistent to stay state_dict-compatible with
        # sae_legacy.py checkpoints; it only affects training, not inference.
        self.register_buffer("mse_scale", torch.ones(()), persistent=False)

        self._init_weights()
        if init_data is not None:
            self.initialize_b_dec_with_data(init_data)
            self.initialize_mse_scale_with_data(init_data)

    def _init_weights(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.w_dec, a=math.sqrt(5))
            # Normalize decoder directions (rows).
            norms = self.w_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.w_dec.data.div_(norms)
            if not self.weight_tying:
                self.w_enc.data.copy_(self.w_dec.data)

    @property
    def _w_enc(self) -> torch.Tensor:
        # In tied mode, encoder uses decoder weights directly.
        return self.w_dec if self.weight_tying else self.w_enc

    @torch.no_grad()
    def initialize_b_dec_with_data(self, data: torch.Tensor):
        """Set b_dec to the mean of the provided activation vectors."""
        self.b_dec.data.copy_(data.float().mean(dim=0))

    @torch.no_grad()
    def initialize_mse_scale_with_data(self, data: torch.Tensor):
        """
        Set mse_scale = 1 / Var(x) from a data sample, so that
        mse_scale * MSE(b_dec-baseline, x) ~= 1 (the official mse_scale).
        """
        baseline = data.float().var(dim=0, unbiased=False).mean()
        self.mse_scale.fill_(1.0 / baseline.clamp_min(1e-12).item())

    @torch.no_grad()
    def normalize_decoder_weights(self):
        norms = self.w_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.w_dec.data.div_(norms)

    @torch.no_grad()
    def remove_parallel_decoder_grads(self):
        """
        Project out the gradient component parallel to each decoder direction
        (official unit_norm_decoder_grad_adjustment_), so optimizer steps stay
        tangent to the unit-norm constraint. Assumes decoder rows are unit
        norm; call normalize_decoder_weights() first. Skipped in tied mode,
        where w_dec doubles as the encoder and its gradient must stay intact.
        """
        if self.weight_tying or self.w_dec.grad is None:
            return
        parallel = (self.w_dec.grad * self.w_dec).sum(dim=1, keepdim=True)
        self.w_dec.grad.sub_(parallel * self.w_dec)

    def scaled_mse(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction MSE scaled to fraction-of-variance-unexplained units."""
        return self.mse_scale * F.mse_loss(x_hat, x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def loss_function(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VanillaSAE(SAE):
    def __init__(self, d_model: int, expansion: int = 4, weight_tying: bool = False, l1_coeff: float = 1e-3, init_data: Optional[torch.Tensor] = None):
        super().__init__(d_model, expansion, weight_tying, init_data)
        self.l1_coeff = l1_coeff

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Center inputs with decoder bias so sparse features model residual structure.
        x_centered = x - self.b_dec
        # Linear projection followed by ReLU activation.
        z = F.linear(x_centered, self._w_enc, self.b_enc)
        z = F.relu(z)
        return z

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    def sparsity_loss(self, encoded: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Normalized L1 (official loss.py): per-sample L1 divided by ||x||_2,
        # so the penalty is scale-free like the normalized reconstruction term.
        return (encoded.abs().sum(dim=-1) / x.norm(dim=-1).clamp_min(1e-12)).mean()

    def loss_function(self, x_hat: torch.Tensor, x: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        return self.scaled_mse(x_hat, x) + self.l1_coeff * self.sparsity_loss(encoded, x)


class TopKSAE(SAE):
    """
    Sparse Autoencoder with Top-K structural sparsity.

    Optionally adds an auxiliary dead-feature loss (use_aux_loss=True, the default)
    that provides gradient signal to encoder directions that have not fired recently,
    preventing feature death over long training runs.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        k: Optional[int] = None,
        weight_tying: bool = False,
        use_aux_loss: bool = False,
        k_aux: Optional[int] = None,
        aux_alpha: Optional[float] = None,
        dead_rate_threshold: Optional[float] = None,
        ema_decay: Optional[float] = None,
        init_data: Optional[torch.Tensor] = None,
    ):
        super().__init__(d_model, expansion, weight_tying, init_data)
        self.k = k if k is not None else max(1, (d_model * expansion) // 200)
        if not 1 <= self.k <= self.d_hidden:
            raise ValueError(
                f"TopK k={self.k} must be in [1, d_hidden={self.d_hidden}]."
            )
        self.register_buffer("k_buffer", torch.tensor(self.k, dtype=torch.long))
        self.use_aux_loss = use_aux_loss
        if use_aux_loss:
            self.k_aux = k_aux if k_aux is not None else min(2 * self.k, self.d_hidden)
            self.aux_alpha = aux_alpha if aux_alpha is not None else 1 / 32
            self.dead_rate_threshold = (
                _DEAD_RATE_THRESHOLD
                if dead_rate_threshold is None
                else dead_rate_threshold
            )
            self.ema_decay = ema_decay if ema_decay is not None else 0.99
            # EMA of per-feature activation rate; starts at 0 so all features are
            # initially considered dead and decay up as they begin to fire.
            self.register_buffer("activation_rate", torch.zeros(self.d_hidden))

    def _pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        # Center inputs with decoder bias so sparse features model residual
        # structure, then apply the encoder linear and ReLU nonlinearity.
        x_centered = x - self.b_dec
        return F.relu(F.linear(x_centered, self._w_enc, self.b_enc))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # 1-2) Encoder pre-activations (centered input -> linear -> ReLU).
        pre_acts = self._pre_acts(x)

        # 3) Keep only top-k activations per sample.
        topk = torch.topk(pre_acts, self.k, dim=-1)

        # 4) Build sparse activation tensor by scattering selected values.
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, index=topk.indices, src=topk.values)
        return acts

    @torch.no_grad()
    def update_dead_features(self, acts: torch.Tensor):
        """Update per-feature activation rate with an EMA over batches."""
        batch_rate = (acts > 0).float().mean(dim=0)  # (d_hidden,) fraction that fired
        self.activation_rate.mul_(self.ema_decay).add_(batch_rate * (1 - self.ema_decay))

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        if self.use_aux_loss and self.training:
            self.update_dead_features(encoded)
        return encoded, decoded

    def _aux_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Aux loss (Appendix A of 'Scaling and Evaluating SAEs', OpenAI 2024).

        The reconstruction residual r = x - x_hat is modelled using the top-k_aux
        dead features, taken from the same encoder activations computed on x. Those
        dead features decode an estimate of r, and the loss penalises how poorly
        they reconstruct it, giving dead directions a targeted gradient signal.

        Following the official implementation, the aux MSE is normalized by the
        per-batch variance of the residual (their normalized_mse), so the term
        stays on the same scale-free footing as the normalized recon loss:

            L_aux = aux_alpha * MSE(decode(z_aux), r) / Var(r)

        where z_aux has top-k_aux non-zero entries drawn only from dead features.
        """
        dead_mask = self.activation_rate < self.dead_rate_threshold  # (d_hidden,)
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return x.new_tensor(0.0)

        # Reuse the encoder activations on x (centered, consistent with encode),
        # restricted to dead features, so they learn what live features missed.
        pre_acts = self._pre_acts(x)
        pre_acts_dead = pre_acts[:, dead_mask]  # (B, num_dead)
        k_aux = min(self.k_aux, num_dead)
        topk_aux = torch.topk(pre_acts_dead, k_aux, dim=-1)
        acts_aux = torch.zeros_like(pre_acts_dead)
        acts_aux.scatter_(-1, topk_aux.indices, topk_aux.values)

        # Reconstruction target is the residual the live features didn't explain.
        # b_dec is already absorbed here (x_hat = ... + b_dec), so x_hat_aux below
        # decodes the dead directions WITHOUT re-adding b_dec.
        residual = (x - x_hat).detach()
        w_dec_dead = self.w_dec[dead_mask]  # (num_dead, d_model)
        x_hat_aux = acts_aux @ w_dec_dead   # (B, d_model)

        residual_var = F.mse_loss(
            residual.mean(dim=0).expand_as(residual), residual
        ).clamp_min(1e-12)
        return self.aux_alpha * F.mse_loss(x_hat_aux, residual) / residual_var

    def loss_function(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        recon_loss = self.scaled_mse(x_hat, x)
        if not self.use_aux_loss:
            return recon_loss
        return recon_loss + self._aux_loss(x, x_hat)


class JumpReLUStepSTEFunction(torch.autograd.Function):
    """
    JumpReLU activation with a Step straight-through estimator (STE):
        f(x, θ) = x · H(x - θ), where H is the Heaviside step.

    Gradients:
        - pre_acts : straight-through (gradient passes only where x > θ).
        - threshold: pseudo-gradient via a rectangular kernel approximation of δ(x - θ):
            ∂/∂θ [x · H(x-θ)] = -x · δ(x-θ) ≈ -x · 1[|x-θ| < bw/2] / bw
    """

    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        return pre_acts * (pre_acts > threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        active = (pre_acts > threshold).float()
        grad_pre_acts = grad_output * active
        in_bandwidth = ((pre_acts - threshold).abs() < bandwidth / 2).float()
        per_sample = grad_output * pre_acts * in_bandwidth / bandwidth
        grad_threshold = -reduce(per_sample, "batch d_hidden -> d_hidden", "sum")
        return grad_pre_acts, grad_threshold, None


class L0StepSTEFunction(torch.autograd.Function):
    """
    Step estimator H(x - θ) used to compute the L0 sparsity penalty.

    Gradient only flows to threshold (pseudo-gradient):
        ∂/∂θ [H(x-θ)] = −δ(x-θ) ≈ -1[|x-θ| < bw/2] / bw
    """

    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        return (pre_acts > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        in_bandwidth = ((pre_acts - threshold).abs() < bandwidth / 2).float()
        per_sample = grad_output * in_bandwidth / bandwidth
        grad_threshold = -reduce(per_sample, "batch d_hidden -> d_hidden", "sum")
        return torch.zeros_like(pre_acts), grad_threshold, None


class JumpReluSAE(SAE):
    """
    Sparse Autoencoder with JumpReLU activation and an L0 sparsity penalty.

    Note: the reconstruction term is variance-normalized like the other SAEs,
    so sparsity_lambda trades off against a loss of order ~1 regardless of the
    raw activation scale. The threshold itself still operates on raw pre-acts,
    so threshold_init remains scale-sensitive.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        threshold_init: float = 0.1,
        bandwidth: float = 0.001,
        sparsity_lambda: float = 1e-4,
        weight_tying: bool = False,
        init_data: Optional[torch.Tensor] = None,
    ):
        super().__init__(d_model, expansion, weight_tying, init_data)
        self.bandwidth = bandwidth
        self.sparsity_lambda = sparsity_lambda
        self.log_threshold = nn.Parameter(
            torch.full((self.d_hidden,), math.log(threshold_init))
        )

    def _encode_with_pre_acts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_dec
        pre_acts = F.linear(x_centered, self._w_enc, self.b_enc)
        threshold = torch.exp(self.log_threshold)
        encoded = JumpReLUStepSTEFunction.apply(pre_acts, threshold, self.bandwidth)
        return encoded, pre_acts

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self._encode_with_pre_acts(x)
        return encoded

    def forward(self, x: torch.Tensor):
        encoded, pre_acts = self._encode_with_pre_acts(x)
        decoded = self.decode(encoded)
        return encoded, decoded, pre_acts

    def sparsity_penalty(self, pre_acts: torch.Tensor) -> torch.Tensor:
        threshold = torch.exp(self.log_threshold)
        active = L0StepSTEFunction.apply(pre_acts, threshold, self.bandwidth)
        l0 = active.sum(dim=-1).mean()
        return self.sparsity_lambda * l0

    def loss_function(self, x_hat: torch.Tensor, x: torch.Tensor, pre_acts: torch.Tensor) -> torch.Tensor:
        return self.scaled_mse(x_hat, x) + self.sparsity_penalty(pre_acts)
