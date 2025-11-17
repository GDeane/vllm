# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os

import pytest
import torch
import torch.nn as nn

from vllm.prefill import ffn_chunk


@pytest.fixture(autouse=True)
def _reset_ffn_chunk_state():
    prev = os.environ.get("VLLM_PREFILL_FFN_CHUNK_SIZE")
    ffn_chunk._STATE.enabled = False
    ffn_chunk._STATE.chunk_size = 0
    yield
    if prev is not None:
        os.environ["VLLM_PREFILL_FFN_CHUNK_SIZE"] = prev
    elif "VLLM_PREFILL_FFN_CHUNK_SIZE" in os.environ:
        del os.environ["VLLM_PREFILL_FFN_CHUNK_SIZE"]
    ffn_chunk._STATE.enabled = False
    ffn_chunk._STATE.chunk_size = 0


class TrackingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.linear(x)


class PrefillChunkable(nn.Module):
    prefill_chunkable = True

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.linear(x)


def test_ffn_chunking_wraps_modules_by_name_or_flag():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = TrackingMLP()
            self.other = PrefillChunkable()

    os.environ["VLLM_PREFILL_FFN_CHUNK_SIZE"] = "4"
    model = Model()
    wrapped = ffn_chunk.enable_prefill_ffn_chunking(model)
    assert wrapped == 2
    assert model.mlp._prefill_ffn_chunk_wrapped
    assert model.other._prefill_ffn_chunk_wrapped


def test_ffn_chunking_splits_sequence_and_matches_reference():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = TrackingMLP()

    torch.manual_seed(1234)
    reference = Model()
    chunked = Model()
    chunked.load_state_dict(reference.state_dict())
    x = torch.randn(1, 6, 4)
    with torch.no_grad():
        expected = reference.mlp(x)

    os.environ["VLLM_PREFILL_FFN_CHUNK_SIZE"] = "2"
    ffn_chunk.enable_prefill_ffn_chunking(chunked)
    with torch.no_grad():
        actual = chunked.mlp(x)

    assert torch.allclose(actual, expected)
    assert chunked.mlp.forward_calls == math.ceil(6 / 2)


def test_ffn_chunking_disabled_for_non_positive_chunk_size():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = TrackingMLP()

    os.environ["VLLM_PREFILL_FFN_CHUNK_SIZE"] = "0"
    model = Model()
    wrapped = ffn_chunk.enable_prefill_ffn_chunking(model)
    assert wrapped == 0
    assert not ffn_chunk._STATE.enabled
    assert not hasattr(model.mlp, "_prefill_ffn_chunk_wrapped")
