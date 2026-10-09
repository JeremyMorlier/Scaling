"""Closed-form cost model: FLOPs and training memory without running the model.

``torchinfo`` and ``FlopCounterMode`` need a real forward pass (hence a GPU big
enough to hold it).  Everything here is arithmetic on the config, so the cost of
a configuration that would OOM is just as cheap to obtain as one that fits --
which is what an extrapolated scaling law needs.

Three families of number are produced, all per image unless stated otherwise:

*FLOPs* -- matmul-class MACs only (conv, linear, attention), the convention
behind the "4.1 GMACs for ResNet-50" figure and the one ``FlopCounterMode``
uses; normalisations and activations are reported separately as ``elem_flops``
because they cost bandwidth, not multipliers.  The backward pass of a weighted
op costs twice its forward (one matmul for the input gradient, one for the
weight gradient), except at the first layer, whose input gradient is never
needed -- so a training step is very nearly, but not exactly, ``3 x`` forward.

*Saved activations* -- the tensors the autograd graph keeps alive from the
forward until the backward consumes them.  This is the term that dominates
training memory and the one a parameter count cannot predict.  It is counted
per *distinct storage*: ``relu(inplace=True)`` writes into the batch-norm
output and saves that one buffer, ``q/k/v`` are views of a single ``qkv``
tensor, and a ``reshape`` of a transposed SDPA output is free.  The accounting
below was checked op by op against ``saved_tensors_hooks`` (see ``--verify``).

*Training memory* -- weights + gradients + optimizer state + batch-norm
buffers + saved activations.  Each term is exact; their sum is a model of the
peak, not a bound on it.  Two effects push the real ``max_memory_allocated``
either way: the backward holds the gradient of whatever activation is live at
the moment, plus cuDNN workspaces and allocator fragmentation (peak up), while
``zero_grad(set_to_none=True)`` frees the gradients before the forward, so the
full activation set and the full gradient buffer never actually coexist (peak
down).  Against the A100 sweep in ``sweep.csv`` the model lands within a few
percent (``--compare sweep.csv``), erring low only for configurations small
enough that the CUDA context dominates.

Usage::

    python -m scaling.analytic --model resnet50 --batch-size 64
    python -m scaling.analytic --model vit_small --set depth=24 --table
    python -m scaling.analytic --model resnet50 --verify     # vs. torch
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

if __package__ in (None, ""):
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "scaling"

from .models import MODEL_AXES
from .models.resnet import ResNetConfig, make_divisible
from .models.vit import ViTConfig

#: Bytes per element.  ``fp32`` is what the sweep measures (``--amp none``).
DTYPE_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "int64": 8}

#: Extra copies of the parameter vector an optimizer keeps.  SGD without
#: momentum is stateless; momentum adds one buffer; Adam keeps two moments.
OPTIMIZER_STATES = {"sgd": 0, "sgd_momentum": 1, "adam": 2, "adamw": 2}

MIB = 2 ** 20


# --------------------------------------------------------------------------- #
# the op record
# --------------------------------------------------------------------------- #

@dataclass
class Op:
    """One primitive of the forward pass and everything it costs.

    Attributes:
        name: dotted module path, as ``torchinfo`` would print it.
        kind: ``conv`` / ``linear`` / ``attn`` / ``norm`` / ``act`` / ``pool``.
        params: trainable parameters.
        buffers: non-trainable state (batch-norm running statistics).
        macs: multiply-accumulates per image, forward, matmul-class only.
        elem_flops: per-image elementwise FLOPs (norms, GELU, residual adds),
            counted apart because they never enter a quoted "GFLOPs" figure.
        saved: activation elements per image retained for the backward pass,
            attributed to the op that *produced* the tensor.
        saved_index: idem, but int64 (max-pool argmax indices).
        saved_const: elements retained that do not scale with the batch
            (batch-norm ``save_mean`` / ``save_invstd``).
        out_elems: output elements per image, used for the inference working set.
        input_grad: whether the backward has to produce a gradient w.r.t. the
            input.  False only for the layer that touches the image.
    """

    name: str
    kind: str
    params: int = 0
    buffers: int = 0
    macs: int = 0
    elem_flops: int = 0
    saved: int = 0
    saved_index: int = 0
    saved_const: int = 0
    out_elems: int = 0
    input_grad: bool = True

    @property
    def weighted(self) -> bool:
        return self.kind in ("conv", "linear")

    @property
    def bwd_macs(self) -> int:
        """Backward MACs: input gradient + weight gradient, each ~ forward."""
        if self.kind == "attn":  # dQ, dK, dV vs. the two forward matmuls
            return 2 * self.macs
        if not self.weighted:
            return 0
        return self.macs * ((1 if self.input_grad else 0) + 1)


# --------------------------------------------------------------------------- #
# memory + cost roll-up
# --------------------------------------------------------------------------- #

@dataclass
class MemoryBreakdown:
    """Training-step memory in bytes, by term."""

    params: int
    grads: int
    optimizer: int
    buffers: int
    activations: int
    activation_indices: int
    inputs: int

    @property
    def weights_total(self) -> int:
        return self.params + self.grads + self.optimizer + self.buffers

    @property
    def activations_total(self) -> int:
        return self.activations + self.activation_indices + self.inputs

    @property
    def total(self) -> int:
        return self.weights_total + self.activations_total

    def as_dict(self) -> dict[str, float]:
        """Every term in bytes, plus a ``_mib`` twin of each for reading."""
        out = dict(self.__dict__) | {"weights_total": self.weights_total,
                                     "activations_total": self.activations_total,
                                     "total": self.total}
        return out | {f"{k}_mib": v / MIB for k, v in out.items()}


@dataclass
class ModelCost:
    """Aggregated cost of one configuration."""

    model: str
    config: dict[str, Any]
    ops: list[Op] = field(default_factory=list)

    # -- structure ---------------------------------------------------------- #
    @property
    def params(self) -> int:
        return sum(op.params for op in self.ops)

    @property
    def buffers(self) -> int:
        return sum(op.buffers for op in self.ops)

    # -- compute ------------------------------------------------------------ #
    @property
    def fwd_macs(self) -> int:
        """Forward MACs per image (the number papers quote as GFLOPs)."""
        return sum(op.macs for op in self.ops)

    @property
    def fwd_flops(self) -> int:
        """Forward FLOPs per image, counting a MAC as two operations."""
        return 2 * self.fwd_macs

    @property
    def bwd_macs(self) -> int:
        return sum(op.bwd_macs for op in self.ops)

    @property
    def train_macs(self) -> int:
        """Forward + backward MACs for one image of one training step."""
        return self.fwd_macs + self.bwd_macs

    @property
    def train_flops(self) -> int:
        return 2 * self.train_macs

    @property
    def elem_flops(self) -> int:
        """Elementwise (norm / activation / residual) FLOPs per image, forward."""
        return sum(op.elem_flops for op in self.ops)

    @property
    def backward_ratio(self) -> float:
        return self.train_macs / self.fwd_macs if self.fwd_macs else float("nan")

    # -- activations -------------------------------------------------------- #
    @property
    def saved_elems(self) -> int:
        """Activation elements per image kept alive for the backward pass."""
        return sum(op.saved for op in self.ops)

    @property
    def saved_index_elems(self) -> int:
        return sum(op.saved_index for op in self.ops)

    @property
    def saved_const_elems(self) -> int:
        return sum(op.saved_const for op in self.ops)

    def saved_bytes(self, batch_size: int, dtype: str = "fp32") -> int:
        """Bytes of saved activations for a batch, indices included."""
        b = DTYPE_BYTES[dtype]
        return (batch_size * self.saved_elems * b
                + batch_size * self.saved_index_elems * DTYPE_BYTES["int64"]
                + self.saved_const_elems * DTYPE_BYTES["fp32"])

    # -- memory ------------------------------------------------------------- #
    def memory(self, batch_size: int, optimizer: str = "sgd_momentum",
               param_dtype: str = "fp32", act_dtype: str = "fp32",
               input_elems: int | None = None) -> MemoryBreakdown:
        """Training-step memory floor, in bytes.

        Args:
            batch_size: images per step.
            optimizer: key of :data:`OPTIMIZER_STATES`.
            param_dtype: storage of weights, gradients and optimizer state.
            act_dtype: storage of saved activations (``bf16`` under autocast).
            input_elems: elements of one input image; defaults to the model's.
        """
        if optimizer not in OPTIMIZER_STATES:
            raise ValueError(f"unknown optimizer {optimizer!r}; "
                             f"expected one of {sorted(OPTIMIZER_STATES)}")
        pb = DTYPE_BYTES[param_dtype]
        p = self.params
        if input_elems is None:
            input_elems = self.config.get("input_elems", 0)
        return MemoryBreakdown(
            params=p * pb,
            grads=p * pb,
            optimizer=OPTIMIZER_STATES[optimizer] * p * pb,
            buffers=self.buffers * DTYPE_BYTES["fp32"],
            activations=(batch_size * self.saved_elems * DTYPE_BYTES[act_dtype]
                         + self.saved_const_elems * DTYPE_BYTES["fp32"]),
            activation_indices=(batch_size * self.saved_index_elems
                                * DTYPE_BYTES["int64"]),
            # The image batch itself: allocated by the loader, saved by the
            # first conv, so it is activation memory that the model does not own.
            inputs=batch_size * input_elems * DTYPE_BYTES[act_dtype],
        )

    def inference_bytes(self, batch_size: int, dtype: str = "fp32") -> int:
        """Weights + the largest live activation pair: a floor, not a model.

        Inference frees every intermediate as soon as its consumer is done, so
        the working set is two adjacent tensors rather than the whole graph.
        Unlike the training figure this one is only a floor: it ignores the
        residual branch a ResNet block holds live, and the measured peak is
        dominated by the cuDNN workspaces ``cudnn.benchmark`` picks.
        """
        b = DTYPE_BYTES[dtype]
        peak = 0
        prev = self.config.get("input_elems", 0)
        for op in self.ops:
            peak = max(peak, prev + op.out_elems)
            prev = op.out_elems
        return (self.params + self.buffers) * DTYPE_BYTES["fp32"] + batch_size * peak * b

    # -- reporting ---------------------------------------------------------- #
    def summary(self, batch_size: int, **memory_kwargs) -> dict[str, Any]:
        mem = self.memory(batch_size, **memory_kwargs)
        return {
            "model": self.model,
            "config": self.config,
            "batch_size": batch_size,
            "params": self.params,
            "buffers": self.buffers,
            "fwd_macs_per_image": self.fwd_macs,
            "fwd_flops_per_image": self.fwd_flops,
            "bwd_macs_per_image": self.bwd_macs,
            "train_macs_per_image": self.train_macs,
            "train_flops_per_image": self.train_flops,
            "train_flops_per_step": self.train_flops * batch_size,
            "backward_ratio": self.backward_ratio,
            "elem_flops_per_image": self.elem_flops,
            "saved_act_elems_per_image": self.saved_elems,
            "saved_act_mib_per_image": self.saved_bytes(1, memory_kwargs.get(
                "act_dtype", "fp32")) / MIB,
            "train_mem": mem.as_dict(),
            "inference_mem_floor_mib": self.inference_bytes(batch_size) / MIB,
        }


# --------------------------------------------------------------------------- #
# shared pieces
# --------------------------------------------------------------------------- #

def _loss_op(num_classes: int) -> Op:
    """Cross-entropy: keeps the log-softmax output and the labels."""
    return Op("loss", "act", elem_flops=3 * num_classes, saved=num_classes,
              saved_index=1)


# --------------------------------------------------------------------------- #
# ResNet-50
# --------------------------------------------------------------------------- #

def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def _conv(name: str, cin: int, cout: int, k: int, s_in: int, stride: int,
          padding: int, input_grad: bool = True) -> tuple[Op, int]:
    """A bias-free conv; returns the op and its output spatial size."""
    s_out = _conv_out(s_in, k, stride, padding)
    out = cout * s_out * s_out
    return Op(name, "conv", params=k * k * cin * cout, macs=k * k * cin * out,
              # Consumed by a batch-norm, which saves its input.
              saved=out, out_elems=out, input_grad=input_grad), s_out


def _bn(name: str, channels: int, spatial: int, saves_output: bool) -> Op:
    """Batch-norm.  ``saves_output`` iff an inplace ReLU follows and keeps it."""
    out = channels * spatial * spatial
    # running_mean + running_var, and the int64 num_batches_tracked scalar.
    return Op(name, "norm", params=2 * channels, buffers=2 * channels + 1,
              # scale + shift, and the mean/var reduction over the batch.
              elem_flops=4 * out, saved=out if saves_output else 0,
              saved_const=2 * channels, out_elems=out)


def _bottleneck(prefix: str, cin: int, planes: int, stride: int, s_in: int,
                downsample: bool) -> tuple[list[Op], int]:
    """One ResNet bottleneck: 1x1 -> 3x3(stride) -> 1x1, plus the shortcut.

    Retained per block: both 1x1/3x3 conv outputs and their post-ReLU
    batch-norm outputs, the expanded conv3 output, and the post-add ReLU output
    (which doubles as the next block's input).  ``bn3`` and the shortcut's
    batch-norm feed only the residual add, which saves nothing, so their
    outputs die immediately.
    """
    cout = planes * 4
    ops: list[Op] = []
    c1, s1 = _conv(f"{prefix}.conv1", cin, planes, 1, s_in, 1, 0)
    ops += [c1, _bn(f"{prefix}.bn1", planes, s1, saves_output=True)]
    c2, s2 = _conv(f"{prefix}.conv2", planes, planes, 3, s1, stride, 1)
    ops += [c2, _bn(f"{prefix}.bn2", planes, s2, saves_output=True)]
    c3, s3 = _conv(f"{prefix}.conv3", planes, cout, 1, s2, 1, 0)
    ops += [c3, _bn(f"{prefix}.bn3", cout, s3, saves_output=False)]
    if downsample:
        cd, sd = _conv(f"{prefix}.downsample.0", cin, cout, 1, s_in, stride, 0)
        ops += [cd, _bn(f"{prefix}.downsample.1", cout, sd, saves_output=False)]
    out = cout * s3 * s3
    # The residual add writes a fresh tensor; the inplace ReLU then saves it.
    ops.append(Op(f"{prefix}.add_relu", "act", elem_flops=2 * out, saved=out,
                  out_elems=out))
    return ops, s3


def resnet50_cost(width_mult: float = 1.0, resolution: int = 224,
                  num_classes: int = 1000, base_width: int = 64,
                  channel_divisor: int = 8, blocks: tuple[int, ...] = (3, 4, 6, 3),
                  ) -> ModelCost:
    """Cost of :class:`scaling.models.resnet.ResNet50` at this configuration."""
    cfg = ResNetConfig(width_mult=width_mult, resolution=resolution,
                       num_classes=num_classes, base_width=base_width,
                       channel_divisor=channel_divisor)
    stem = make_divisible(base_width * width_mult, channel_divisor)
    planes = [make_divisible(base_width * (2 ** i) * width_mult, channel_divisor)
              for i in range(4)]

    ops: list[Op] = []
    # Stem.  The image gradient is never needed, so conv1 backward is half price.
    c1, s = _conv("conv1", 3, stem, 7, resolution, 2, 3, input_grad=False)
    ops += [c1, _bn("bn1", stem, s, saves_output=True)]
    s_pool = _conv_out(s, 3, 2, 1)
    pooled = stem * s_pool * s_pool
    ops.append(Op("maxpool", "pool", elem_flops=9 * pooled, saved=pooled,
                  saved_index=pooled, out_elems=pooled))

    cin = stem
    for stage, (p, n) in enumerate(zip(planes, blocks), start=1):
        for i in range(n):
            stride = 2 if (i == 0 and stage > 1) else 1
            block_ops, s_pool = _bottleneck(
                f"layer{stage}.{i}", cin, p, stride, s_pool,
                downsample=(i == 0 and (stride != 1 or cin != p * 4)))
            ops += block_ops
            cin = p * 4

    ops.append(Op("avgpool", "pool", elem_flops=cin * s_pool * s_pool,
                  saved=cin, out_elems=cin))
    ops.append(Op("fc", "linear", params=cin * num_classes + num_classes,
                  macs=cin * num_classes, out_elems=num_classes))
    ops.append(_loss_op(num_classes))

    return ModelCost("resnet50", {
        "width_mult": width_mult, "resolution": resolution,
        "num_classes": num_classes, "stem_width": stem, "planes": planes,
        "input_elems": 3 * resolution * resolution,
        "input_shape": list(cfg.input_shape),
    }, ops)


# --------------------------------------------------------------------------- #
# ViT
# --------------------------------------------------------------------------- #

def vit_cost(embed_dim: int = 384, depth: int = 12, mlp_dim: int = 1536,
             num_patches: int = 196, patch_size: int = 16,
             num_heads: int | None = None, head_dim: int = 64,
             num_classes: int = 1000, qkv_bias: bool = True,
             attn_impl: str = "flash") -> ModelCost:
    """Cost of :class:`scaling.models.vit.VisionTransformer`.

    ``attn_impl`` picks what the attention backward keeps: ``flash`` (what
    ``scaled_dot_product_attention`` selects on an A100) stores only the
    per-row log-sum-exp and recomputes the scores, so memory is linear in the
    sequence length; ``math`` stores the full ``heads x N x N`` probability
    matrix and is quadratic.
    """
    if attn_impl not in ("flash", "math"):
        raise ValueError(f"attn_impl must be 'flash' or 'math', got {attn_impl!r}")
    cfg = ViTConfig(embed_dim=embed_dim, depth=depth, mlp_dim=mlp_dim,
                    num_patches=num_patches, patch_size=patch_size,
                    num_heads=num_heads, head_dim=head_dim,
                    num_classes=num_classes, qkv_bias=qkv_bias)
    d, m, n, h = embed_dim, mlp_dim, cfg.seq_len, cfg.num_heads
    r, p = cfg.resolution, patch_size

    ops: list[Op] = []
    # Patch embedding: a stride-``p`` conv over the image, then cls + position.
    ops.append(Op("patch_embed.proj", "conv", params=3 * p * p * d + d,
                  macs=3 * p * p * d * num_patches, out_elems=num_patches * d,
                  input_grad=False))
    ops.append(Op("pos_embed", "act", params=d + n * d, elem_flops=n * d,
                  # The summed tensor is block 0's input, saved by its norm1.
                  saved=n * d, out_elems=n * d))

    for i in range(depth):
        b = f"blocks.{i}"
        # LayerNorm saves its output for the linear that follows, plus the
        # per-token mean and reciprocal standard deviation.
        ops.append(Op(f"{b}.norm1", "norm", params=2 * d, elem_flops=5 * n * d,
                      saved=n * d + 2 * n, out_elems=n * d))
        ops.append(Op(f"{b}.attn.qkv", "linear",
                      params=3 * d * d + (3 * d if qkv_bias else 0),
                      macs=n * d * 3 * d, saved=3 * n * d, out_elems=3 * n * d))
        # q, k and v are strided views of that one qkv tensor: no copy.
        attn_saved = n * d + (h * n if attn_impl == "flash" else h * n * n)
        ops.append(Op(f"{b}.attn.sdpa", "attn", macs=2 * n * n * d,
                      elem_flops=5 * h * n * n, saved=attn_saved,
                      out_elems=n * d))
        # proj's input is a reshape of the SDPA output, which is already saved;
        # its own output feeds the residual add, which saves nothing.
        ops.append(Op(f"{b}.attn.proj", "linear", params=d * d + d,
                      macs=n * d * d, out_elems=n * d))
        ops.append(Op(f"{b}.add_attn", "act", elem_flops=n * d, saved=n * d,
                      out_elems=n * d))
        ops.append(Op(f"{b}.norm2", "norm", params=2 * d, elem_flops=5 * n * d,
                      saved=n * d + 2 * n, out_elems=n * d))
        ops.append(Op(f"{b}.mlp.fc1", "linear", params=d * m + m, macs=n * d * m,
                      saved=n * m, out_elems=n * m))
        ops.append(Op(f"{b}.mlp.act", "act", elem_flops=8 * n * m, saved=n * m,
                      out_elems=n * m))
        ops.append(Op(f"{b}.mlp.fc2", "linear", params=m * d + d, macs=n * m * d,
                      out_elems=n * d))
        ops.append(Op(f"{b}.add_mlp", "act", elem_flops=n * d, saved=n * d,
                      out_elems=n * d))

    ops.append(Op("norm", "norm", params=2 * d, elem_flops=5 * n * d,
                  saved=n * d + 2 * n, out_elems=n * d))
    # The head reads the class token only: a view of the tensor norm already saved.
    ops.append(Op("head", "linear", params=d * num_classes + num_classes,
                  macs=d * num_classes, out_elems=num_classes))
    ops.append(_loss_op(num_classes))

    return ModelCost("vit_small", {
        "embed_dim": d, "depth": depth, "mlp_dim": m, "num_patches": num_patches,
        "seq_len": n, "num_heads": h, "patch_size": p, "resolution": r,
        "num_classes": num_classes, "attn_impl": attn_impl,
        "input_elems": 3 * r * r, "input_shape": list(cfg.input_shape),
    }, ops)


def model_cost(name: str, num_classes: int = 1000, **params) -> ModelCost:
    """Analytic twin of :func:`scaling.models.build_model`."""
    if name not in MODEL_AXES:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODEL_AXES)}")
    config = dict(MODEL_AXES[name])
    extra = {k: params.pop(k) for k in ("attn_impl",) if k in params}
    unknown = set(params) - set(config)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)} for {name}; "
            f"valid axes are {sorted(config)}")
    config.update(params)
    if name == "resnet50":
        return resnet50_cost(num_classes=num_classes, **config)
    return vit_cost(num_classes=num_classes, **config, **extra)


# --------------------------------------------------------------------------- #
# verification against torch
# --------------------------------------------------------------------------- #

def measured_cost(name: str, batch_size: int = 2, device: str = "cpu",
                  num_classes: int = 1000, **params) -> dict[str, Any]:
    """Ground truth for :func:`model_cost`, obtained by building the model.

    Parameters come from the module itself, MACs from ``FlopCounterMode`` and
    saved activations from ``saved_tensors_hooks``, deduplicated by storage so
    that views and inplace writes are counted once -- which is what the
    allocator sees.  Needs torch, and enough memory for one forward/backward.
    """
    import torch
    import torch.nn as nn

    from .models import build_model

    dev = torch.device(device)
    model = build_model(name, num_classes=num_classes, **params).to(dev)
    shape = (batch_size, *model.config.input_shape)
    x = torch.randn(*shape, device=dev)
    targets = torch.randint(0, num_classes, (batch_size,), device=dev)

    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x[:1])
    macs = counter.get_total_flops() / 2
    # torch's CPU flash-attention op is absent from the counter's registry, so
    # on CPU the attention matmuls silently count as zero.  Say so rather than
    # letting the caller read a 4% gap as a modelling error.
    counted = counter.get_flop_counts().get("Global", {})
    sdpa_counted = any("scaled_dot_product" in str(k) for k in counted)

    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    param_storages |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    seen: set[int] = set()
    saved_bytes = 0

    def pack(t: "torch.Tensor") -> "torch.Tensor":
        nonlocal saved_bytes
        # 0-d tensors are bookkeeping, not activations: cross-entropy's
        # total_weight, and the philox seed/offset some torch versions attach
        # to flash attention.  Counting them would make the comparison track
        # the torch version instead of the architecture.
        key = t.untyped_storage().data_ptr()
        if t.dim() and key not in seen and key not in param_storages:
            seen.add(key)
            saved_bytes += t.untyped_storage().nbytes()
        return t

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        nn.functional.cross_entropy(model(x), targets)

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "buffers": sum(b.numel() for b in model.buffers()),
        "fwd_macs_per_image": macs,
        "saved_act_bytes": saved_bytes,
        "batch_size": batch_size,
        "sdpa_counted": sdpa_counted,
    }


def verify(name: str, batch_size: int = 2, device: str = "cpu",
           num_classes: int = 1000, **params) -> dict[str, dict[str, Any]]:
    """Compare :func:`model_cost` with :func:`measured_cost`, term by term."""
    cost = model_cost(name, num_classes=num_classes, **params)
    truth = measured_cost(name, batch_size=batch_size, device=device,
                          num_classes=num_classes, **params)
    mem = cost.memory(batch_size)
    predicted = {
        "params": cost.params,
        "buffers": cost.buffers,
        "fwd_macs_per_image": cost.fwd_macs,
        "saved_act_bytes": mem.activations_total,
    }
    notes: list[str] = []
    attn = sum(op.macs for op in cost.ops if op.kind == "attn")
    if attn and not truth.pop("sdpa_counted", True):
        predicted["fwd_macs_per_image"] -= attn
        notes.append(
            f"torch does not register scaled_dot_product_attention on {device}; "
            f"{attn} attention MACs/image excluded from the MAC comparison")
    truth.pop("sdpa_counted", None)
    out: dict[str, dict[str, Any]] = {}
    for key, want in truth.items():
        if key not in predicted:
            continue
        got = predicted[key]
        out[key] = {"analytic": got, "torch": want,
                    "rel_err": (got - want) / want if want else 0.0,
                    "exact": got == want}
    if notes:
        out["notes"] = notes
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _si(value: float, unit: str = "") -> str:
    """4089184256 -> '4.089 G'."""
    for scale, suffix in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"),
                          (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:8.3f} {suffix}{unit}"
    return f"{value:8.3f}  {unit}"


def format_table(cost: ModelCost, batch_size: int, act_dtype: str = "fp32",
                 group: bool = False) -> str:
    """A per-op table in the shape ``torchinfo`` prints."""
    rows = cost.ops
    if group:
        merged: dict[str, Op] = {}
        for op in rows:
            # layer3.4.conv2 -> layer3, blocks.7.mlp.fc1 -> blocks.7
            parts = op.name.split(".")
            key = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
            acc = merged.setdefault(key, Op(key, "group"))
            acc.params += op.params
            acc.buffers += op.buffers
            acc.macs += op.macs
            acc.elem_flops += op.elem_flops
            acc.saved += op.saved
            acc.saved_index += op.saved_index
            acc.saved_const += op.saved_const
        rows = list(merged.values())

    b = DTYPE_BYTES[act_dtype]
    head = (f"{'layer':<28}{'kind':<8}{'params':>14}{'MACs/img':>16}"
            f"{'saved/img':>14}{'saved MiB @B=' + str(batch_size):>20}")
    lines = [head, "-" * len(head)]
    for op in rows:
        saved_b = batch_size * (op.saved * b + op.saved_index * 8)
        lines.append(
            f"{op.name:<28}{op.kind:<8}{op.params:>14,}{op.macs:>16,}"
            f"{op.saved + op.saved_index:>14,}{saved_b / MIB:>20,.2f}")
    lines.append("-" * len(head))
    total_b = cost.saved_bytes(batch_size, act_dtype)
    lines.append(
        f"{'TOTAL':<28}{'':<8}{cost.params:>14,}{cost.fwd_macs:>16,}"
        f"{cost.saved_elems + cost.saved_index_elems:>14,}{total_b / MIB:>20,.2f}")
    return "\n".join(lines)


def format_summary(cost: ModelCost, batch_size: int,
                   optimizer: str = "sgd_momentum", param_dtype: str = "fp32",
                   act_dtype: str = "fp32", images: int | None = None) -> str:
    """The headline numbers: FLOPs, then the training-memory breakdown."""
    mem = cost.memory(batch_size, optimizer=optimizer, param_dtype=param_dtype,
                      act_dtype=act_dtype)
    cfg = " ".join(f"{k}={v}" for k, v in cost.config.items()
                   if k in ("width_mult", "resolution", "embed_dim", "depth",
                            "mlp_dim", "num_patches", "seq_len", "num_heads"))
    shape = "x".join(str(s) for s in cost.config["input_shape"])
    out = [
        f"{cost.model}  {cfg}",
        f"input {batch_size}x{shape}   optimizer={optimizer}  "
        f"params={param_dtype} activations={act_dtype}",
        "",
        f"  parameters          {cost.params:>15,}   "
        f"{cost.params * DTYPE_BYTES[param_dtype] / MIB:8.2f} MiB",
        f"  buffers (BN)        {cost.buffers:>15,}   "
        f"{cost.buffers * 4 / MIB:8.2f} MiB",
        "",
        "compute, per image",
        f"  forward             {_si(cost.fwd_macs, 'MACs')}   "
        f"{_si(cost.fwd_flops, 'FLOPs')}",
        f"  backward            {_si(cost.bwd_macs, 'MACs')}   "
        f"{_si(2 * cost.bwd_macs, 'FLOPs')}",
        f"  training step       {_si(cost.train_macs, 'MACs')}   "
        f"{_si(cost.train_flops, 'FLOPs')}   ({cost.backward_ratio:.2f}x forward)",
        f"  elementwise (fwd)   {_si(cost.elem_flops, 'FLOPs')}   "
        "norms, activations, residual adds; excluded above",
        "",
        f"compute, per step of {batch_size}",
        f"  forward             {_si(cost.fwd_flops * batch_size, 'FLOPs')}",
        f"  training step       {_si(cost.train_flops * batch_size, 'FLOPs')}",
    ]
    if images:
        out += [f"  {images:,} images      "
                f"{_si(cost.train_flops * images, 'FLOPs')} to train"]
    out += [
        "",
        f"training memory, batch {batch_size}",
        f"  parameters          {mem.params / MIB:10.2f} MiB",
        f"  gradients           {mem.grads / MIB:10.2f} MiB",
        f"  optimizer state     {mem.optimizer / MIB:10.2f} MiB   "
        f"({OPTIMIZER_STATES[optimizer]} x params)",
        f"  BN buffers          {mem.buffers / MIB:10.2f} MiB",
        f"  saved activations   {mem.activations / MIB:10.2f} MiB   "
        f"({cost.saved_elems:,} elems/img)",
        f"  argmax indices      {mem.activation_indices / MIB:10.2f} MiB",
        f"  input batch         {mem.inputs / MIB:10.2f} MiB",
        f"  {'-' * 34}",
        f"  weights + states    {mem.weights_total / MIB:10.2f} MiB",
        f"  activations         {mem.activations_total / MIB:10.2f} MiB",
        f"  total               {mem.total / MIB:10.2f} MiB",
        "",
        f"inference memory floor  {cost.inference_bytes(batch_size, act_dtype) / MIB:10.2f} MiB"
        "   weights + largest live pair",
        "",
        "Every term is exact; the total models the peak rather than bounding "
        "it.  The backward",
        "also holds activation gradients and cuDNN workspaces, while "
        "zero_grad(set_to_none=True)",
        "means the gradients are not live during the forward.  --compare "
        "sweep.csv puts the model",
        "next to the measurements.",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# the whole sweep, analytically
# --------------------------------------------------------------------------- #

#: Columns of the analytic sweep CSV.  The identity columns are spelled exactly
#: as :mod:`scaling.aggregate` spells them, so the analytic and measured tables
#: line up run for run and plot through the same code.
SWEEP_COLUMNS = [
    "run_id", "model", "axis", "swept_value", "status", "batch_size",
    "params", "resolution", "seq_len", "num_heads",
    "fwd_macs_per_image", "fwd_gflops_per_image", "bwd_gflops_per_image",
    "train_gflops_per_image", "train_gflops_per_step",
    "saved_act_elems_per_image", "act_mem_mib", "weights_mem_mib",
    "train_mem_mib", "infer_mem_floor_mib",
    "optimizer", "param_dtype", "act_dtype", "source",
]


def sweep_rows(models: list[str] | None = None, batch_size: int = 64,
               optimizer: str = "sgd_momentum", param_dtype: str = "fp32",
               act_dtype: str = "fp32", attn_impl: str = "flash",
               num_classes: int = 1000) -> list[dict[str, Any]]:
    """Cost every configuration of the one-factor-at-a-time grid.

    The grid comes from :func:`scaling.sweep.generate`, the same one the
    measured sweep walks, so run ids match and the two tables are directly
    comparable.  To cost configurations no GPU here could hold, widen
    ``SWEEP_VALUES`` in :mod:`scaling.sweep`: nothing is executed, so the extra
    points are free.
    """
    from .sweep import generate

    rows: list[dict[str, Any]] = []
    for run in generate(models=models, batch_size=batch_size):
        params = dict(run["params"])
        if run["model"] == "vit_small":
            params["attn_impl"] = attn_impl
        cost = model_cost(run["model"], num_classes=num_classes, **params)
        mem = cost.memory(batch_size, optimizer=optimizer,
                          param_dtype=param_dtype, act_dtype=act_dtype)
        axis = run["axis"]
        rows.append({
            "run_id": run["run_id"],
            "model": run["model"],
            "axis": axis,
            # As in the measured CSV: at the base point every axis sits at its
            # base value, so there is no single swept value to report.
            "swept_value": run["params"].get(axis) if axis != "base" else "",
            "status": "ok",
            "batch_size": batch_size,
            "params": cost.params,
            "resolution": cost.config["resolution"],
            "seq_len": cost.config.get("seq_len", ""),
            "num_heads": cost.config.get("num_heads", ""),
            "fwd_macs_per_image": cost.fwd_macs,
            "fwd_gflops_per_image": cost.fwd_flops / 1e9,
            "bwd_gflops_per_image": 2 * cost.bwd_macs / 1e9,
            "train_gflops_per_image": cost.train_flops / 1e9,
            "train_gflops_per_step": cost.train_flops * batch_size / 1e9,
            "saved_act_elems_per_image": cost.saved_elems,
            "act_mem_mib": mem.activations_total / MIB,
            "weights_mem_mib": mem.weights_total / MIB,
            "train_mem_mib": mem.total / MIB,
            "infer_mem_floor_mib": cost.inference_bytes(batch_size, act_dtype) / MIB,
            "optimizer": optimizer,
            "param_dtype": param_dtype,
            "act_dtype": act_dtype,
            "source": "analytic",
        })
    return rows


def write_sweep_csv(path: str, rows: list[dict[str, Any]]) -> None:
    import csv
    import os

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# comparison against a measured sweep
# --------------------------------------------------------------------------- #

def compare_csv(path: str, optimizer: str = "sgd_momentum") -> str:
    """Put the analytic numbers next to a ``scaling.aggregate`` CSV."""
    import csv

    head = (f"{'run_id':<34}{'params':>12}{'GMACs':>9}{'GMACs':>9}"
            f"{'train mem':>11}{'peak meas.':>11}{'model/meas':>11}")
    lines = [f"{'':<34}{'analytic=torch':>12}{'analytic':>9}{'measured':>9}"
             f"{'MiB':>11}{'MiB':>11}{'':>11}", head, "-" * len(head)]
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            axis, value = row["axis"], row["swept_value"]
            params: dict[str, Any] = {}
            if axis not in ("", "base") and value not in ("", None):
                params[axis] = float(value) if axis == "width_mult" else int(float(value))
            cost = model_cost(row["model"], **params)
            batch = int(row["batch_size"])
            act_dtype = {"none": "fp32"}.get(row.get("amp", "none"), "bf16")
            mem = cost.memory(batch, optimizer=optimizer, act_dtype=act_dtype)
            measured_mem = float(row["train_peak_mem_mib"])
            measured_macs = float(row["fwd_macs_per_image"])
            match = "=" if cost.params == int(row["params"]) else "!"
            lines.append(
                f"{row['run_id']:<34}{cost.params:>11,}{match}"
                f"{cost.fwd_macs / 1e9:>9.3f}{measured_macs / 1e9:>9.3f}"
                f"{mem.total / MIB:>11.1f}{measured_mem:>11.1f}"
                f"{mem.total / MIB / measured_mem:>11.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_kv(items: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        key, _, value = item.partition("=")
        for cast in (int, float):
            try:
                out[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scaling.analytic",
        description="Analytic FLOPs and training memory, no forward pass needed.")
    p.add_argument("--model", choices=sorted(MODEL_AXES),
                   help="architecture to cost (omit with --compare)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="scaling-axis overrides, e.g. --set depth=24 embed_dim=768")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--optimizer", choices=sorted(OPTIMIZER_STATES),
                   default="sgd_momentum",
                   help="how many extra parameter-sized buffers to charge")
    p.add_argument("--param-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--act-dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="bf16/fp16 for an autocast run: saved activations shrink, "
                        "weights and optimizer state do not")
    p.add_argument("--attn-impl", choices=["flash", "math"], default="flash",
                   help="ViT only: what the attention backward keeps")
    p.add_argument("--images", type=int,
                   help="total images seen, to turn the per-step cost into a "
                        "training budget (e.g. 1281167 x 90 epochs)")
    p.add_argument("--table", action="store_true", help="per-op breakdown")
    p.add_argument("--group", action="store_true",
                   help="collapse the table to top-level modules")
    p.add_argument("--json", action="store_true", help="machine-readable summary")
    p.add_argument("--verify", action="store_true",
                   help="check the model against torch on a tiny batch")
    p.add_argument("--verify-batch", type=int, default=2)
    p.add_argument("--verify-device", default="cpu")
    p.add_argument("--compare", metavar="CSV",
                   help="tabulate against a measured sweep CSV")
    p.add_argument("--sweep-csv", metavar="CSV",
                   help="cost the whole one-factor-at-a-time grid into this CSV, "
                        "ready for scaling.plot")
    p.add_argument("--models", nargs="*", choices=sorted(MODEL_AXES),
                   help="restrict --sweep-csv to these architectures")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.compare:
        print(compare_csv(args.compare, optimizer=args.optimizer))
        return 0
    if args.sweep_csv:
        rows = sweep_rows(models=args.models, batch_size=args.batch_size,
                          optimizer=args.optimizer, param_dtype=args.param_dtype,
                          act_dtype=args.act_dtype, attn_impl=args.attn_impl,
                          num_classes=args.num_classes)
        write_sweep_csv(args.sweep_csv, rows)
        print(f"wrote {len(rows)} configurations to {args.sweep_csv}")
        return 0
    if not args.model:
        raise SystemExit("--model is required (or use --compare CSV)")

    params = _parse_kv(args.set)
    if args.model == "vit_small":
        params["attn_impl"] = args.attn_impl
    cost = model_cost(args.model, num_classes=args.num_classes, **params)

    if args.verify:
        report = verify(args.model, args.verify_batch, device=args.verify_device,
                        num_classes=args.num_classes,
                        **{k: v for k, v in params.items() if k != "attn_impl"})
        print(json.dumps(report, indent=2))
        return 0 if all(v["exact"] for k, v in report.items()
                        if k != "notes") else 1

    if args.json:
        print(json.dumps(cost.summary(
            args.batch_size, optimizer=args.optimizer,
            param_dtype=args.param_dtype, act_dtype=args.act_dtype), indent=2))
        return 0

    print(format_summary(cost, args.batch_size, optimizer=args.optimizer,
                         param_dtype=args.param_dtype, act_dtype=args.act_dtype,
                         images=args.images))
    if args.table or args.group:
        print()
        print(format_table(cost, args.batch_size, act_dtype=args.act_dtype,
                           group=args.group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
