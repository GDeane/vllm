# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Utilities to chunk feed-forward (FFN/MLP) modules during hybrid prefilling.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _ChunkState:
    enabled: bool = False
    chunk_size: int = 0


_STATE = _ChunkState()


def enable_prefill_ffn_chunking(model: nn.Module) -> int:
    """Enable chunked FFN execution for hybrid prefill mode."""
    chunk_size = os.getenv("VLLM_PREFILL_FFN_CHUNK_SIZE")
    try:
        chunk_size_val = int(chunk_size) if chunk_size else 2048
    except ValueError:
        logger.warning(
            "Invalid VLLM_PREFILL_FFN_CHUNK_SIZE=%s; falling back to 2048.",
            chunk_size,
        )
        chunk_size_val = 2048

    if chunk_size_val <= 0:
        logger.info(
            "Prefill FFN chunking disabled because chunk size %d <= 0.",
            chunk_size_val,
        )
        return 0

    _STATE.enabled = True
    _STATE.chunk_size = chunk_size_val

    wrapped = 0
    for module in model.modules():
        if _should_wrap_module(module):
            wrapped += int(_wrap_module_forward(module))

    if wrapped == 0:
        logger.warning_once(
            "Prefill FFN chunking requested but no MLP modules were wrapped."
        )
    else:
        logger.info(
            "Prefill FFN chunking enabled for %d modules (chunk size=%d tokens).",
            wrapped,
            chunk_size_val,
        )
    return wrapped


def _should_wrap_module(module: nn.Module) -> bool:
    if getattr(module, "_prefill_ffn_chunk_wrapped", False):
        return False

    name = module.__class__.__name__.lower()
    if "mlp" in name or "ffn" in name:
        return True
    return bool(getattr(module, "prefill_chunkable", False))


def _wrap_module_forward(module: nn.Module) -> bool:
    forward = getattr(module, "forward", None)
    if not callable(forward):
        return False

    @wraps(forward)
    def chunked_forward(*args, **kwargs):
        if not _STATE.enabled:
            return forward(*args, **kwargs)
        return _run_chunked_forward(module, forward, args, kwargs)

    module.forward = chunked_forward  # type: ignore[assignment]
    module._prefill_ffn_chunk_wrapped = True  # type: ignore[attr-defined]
    return True


@dataclass
class _TensorArg:
    kind: str  # "arg" or "kwarg"
    index: int | None
    name: str | None
    tensor: torch.Tensor


def _find_tensor_argument(
    args: Sequence[Any], kwargs: dict[str, Any]
) -> _TensorArg | None:
    start_idx = 0
    if args and isinstance(args[0], nn.Module):
        start_idx = 1

    for idx in range(start_idx, len(args)):
        val = args[idx]
        if torch.is_tensor(val):
            return _TensorArg("arg", idx, None, val)

    for name, val in kwargs.items():
        if torch.is_tensor(val):
            return _TensorArg("kwarg", None, name, val)

    return None


def _infer_chunk_dim(tensor: torch.Tensor) -> int:
    if tensor.dim() <= 1:
        return 0
    if tensor.dim() == 2:
        return 0

    candidate_dims = list(range(tensor.dim() - 1))  # ignore hidden dim
    seq_dim = max(candidate_dims, key=lambda d: tensor.shape[d])
    return seq_dim


def _split_tensor(
    tensor: torch.Tensor, dim: int, chunk_size: int
) -> list[torch.Tensor]:
    if chunk_size >= tensor.shape[dim]:
        return [tensor]
    return list(torch.split(tensor, chunk_size, dim=dim))


def _concat_outputs(outputs: list[Any], dim_hint: int) -> Any:
    first = outputs[0]
    if torch.is_tensor(first):
        dim = dim_hint if dim_hint < first.dim() else 0
        return torch.cat(outputs, dim=dim)
    if isinstance(first, tuple):
        concatenated = []
        for i in range(len(first)):
            components = [out[i] for out in outputs]
            if torch.is_tensor(components[0]):
                dim = dim_hint if dim_hint < components[0].dim() else 0
                concatenated.append(torch.cat(components, dim=dim))
            else:
                concatenated.append(components[-1])
        return tuple(concatenated)
    raise RuntimeError(
        f"Prefill FFN chunking only supports tensor outputs; got {type(first)} instead."
    )


def _run_chunked_forward(
    module: nn.Module,
    forward: Callable,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    tensor_arg = _find_tensor_argument(args, kwargs)
    if tensor_arg is None:
        return forward(*args, **kwargs)

    tensor = tensor_arg.tensor
    if tensor.dim() == 0:
        return forward(*args, **kwargs)

    chunk_dim = _infer_chunk_dim(tensor)
    splits = _split_tensor(tensor, chunk_dim, _STATE.chunk_size)
    if len(splits) == 1:
        return forward(*args, **kwargs)

    args_list = list(args)
    original_tensor = tensor
    outputs = []

    for chunk in splits:
        if tensor_arg.kind == "arg" and tensor_arg.index is not None:
            args_list[tensor_arg.index] = chunk
        elif tensor_arg.kind == "kwarg" and tensor_arg.name is not None:
            kwargs[tensor_arg.name] = chunk

        outputs.append(forward(*args_list, **kwargs))

    # Restore arguments.
    if tensor_arg.kind == "arg" and tensor_arg.index is not None:
        args_list[tensor_arg.index] = original_tensor
    elif tensor_arg.kind == "kwarg" and tensor_arg.name is not None:
        kwargs[tensor_arg.name] = original_tensor

    try:
        return _concat_outputs(outputs, dim_hint=chunk_dim)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning(
            "Prefill FFN chunking failed for module %s (%s). "
            "Falling back to unchunked forward. Error: %s",
            module.__class__.__name__,
            module,
            exc,
        )
        # Run original forward once without chunking.
        return forward(*args, **kwargs)
