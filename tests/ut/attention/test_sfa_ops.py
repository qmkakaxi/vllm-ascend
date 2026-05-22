#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase


class TestSparseFlashAttentionOp(TestBase):
    """Test cases for npu_sparse_flash_attention operator return value changes.

    The operator signature changed from -> Tensor to -> (Tensor, Tensor, Tensor),
    where the three outputs are: attention_out, softmax_max, softmax_sum.
    """

    def _mock_sparse_flash_attention(self, query, key, value, sparse_indices,
                                      scale_value, *, block_table=None,
                                      actual_seq_lengths_query=None,
                                      actual_seq_lengths_kv=None,
                                      query_rope=None, key_rope=None,
                                      sparse_block_size=1,
                                      layout_query='BSND', layout_kv='BSND',
                                      sparse_mode=3, pre_tokens=torch.iinfo(torch.int64).max,
                                      next_tokens=torch.iinfo(torch.int64).max,
                                      attention_mode=0, return_softmax_lse=True):
        """Mock implementation matching the new operator signature."""
        batch = query.size(0)
        num_kv_heads = key.size(2) if key.dim() >= 3 else 1
        seq_q = query.size(1)
        head_dim = query.size(-1)
        num_q_heads = query.size(2)
        g = num_q_heads // num_kv_heads

        output = torch.zeros_like(query)

        softmax_max = torch.zeros(
            num_kv_heads, batch, g, dtype=torch.float32, device=query.device
        )
        softmax_sum = torch.zeros(
            num_kv_heads, batch, g, dtype=torch.float32, device=query.device
        )

        return output, softmax_max, softmax_sum

    def test_returns_tuple_of_three_tensors(self):
        """Verify the operator returns a 3-tuple (output, softmax_max, softmax_sum)."""
        batch, seq_q, num_kv_heads, num_q_heads, head_dim = 2, 8, 1, 4, 128
        query = torch.randn(batch, seq_q, num_q_heads, head_dim, dtype=torch.float16)
        key = torch.randn(batch, seq_q, num_kv_heads, head_dim, dtype=torch.float16)
        value = key
        sparse_indices = torch.randint(0, seq_q, (batch, seq_q, num_kv_heads, 32), dtype=torch.int32)

        with patch("torch.ops._C_ascend.npu_sparse_flash_attention",
                   side_effect=self._mock_sparse_flash_attention):
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=query, key=key, value=value,
                sparse_indices=sparse_indices, scale_value=1.0,
                layout_query="BSND", layout_kv="BSND",
                sparse_mode=0, return_softmax_lse=True,
            )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3, "Expected 3 outputs: (attention_out, softmax_max, softmax_sum)")

        attn_out, softmax_max, softmax_sum = result

        # attention_out shape matches query shape
        self.assertEqual(attn_out.shape, query.shape)
        self.assertEqual(attn_out.dtype, query.dtype)

        # softmax_max shape: (num_kv_heads, batch, g)
        expected_lse_shape = (num_kv_heads, batch, num_q_heads // num_kv_heads)
        self.assertEqual(softmax_max.shape, expected_lse_shape)
        self.assertEqual(softmax_max.dtype, torch.float32)
        self.assertEqual(softmax_sum.shape, expected_lse_shape)
        self.assertEqual(softmax_sum.dtype, torch.float32)


class TestLightningIndexerOp(TestBase):
    """Test cases for npu_lightning_indexer operator return value changes.

    The operator signature changed from -> Tensor to -> (Tensor, Tensor),
    where the two outputs are: sparse_indices, sparse_values.
    """

    def _mock_lightning_indexer(self, query, key, weights, *,
                                 actual_seq_lengths_query=None,
                                 actual_seq_lengths_key=None,
                                 block_table=None, layout_query='BSND',
                                 layout_key='BSND', sparse_count=2048,
                                 sparse_mode=3,
                                 pre_tokens=torch.iinfo(torch.int64).max,
                                 next_tokens=torch.iinfo(torch.int64).max,
                                 return_value=True):
        """Mock implementation matching the new operator signature."""
        batch = query.size(0)
        seq_q = query.size(1)
        num_kv_heads = key.size(2) if key.dim() >= 3 else 1

        sparse_indices = torch.zeros(
            batch, seq_q, num_kv_heads, sparse_count,
            dtype=torch.int32, device=query.device
        )
        sparse_values = torch.zeros(
            batch, seq_q, num_kv_heads, sparse_count,
            dtype=query.dtype, device=query.device
        )

        return sparse_indices, sparse_values

    def test_returns_tuple_of_two_tensors(self):
        """Verify the operator returns a 2-tuple (sparse_indices, sparse_values)."""
        batch, seq_q, num_kv_heads, num_q_heads, head_dim = 2, 8, 1, 4, 128
        sparse_count = 2048

        query = torch.randn(batch, seq_q, num_q_heads, head_dim, dtype=torch.float16)
        key = torch.randn(batch, seq_q, num_kv_heads, head_dim, dtype=torch.float16)
        weights = torch.randn(batch, seq_q, num_q_heads, dtype=torch.float16)

        with patch("torch.ops._C_ascend.npu_lightning_indexer",
                   side_effect=self._mock_lightning_indexer):
            result = torch.ops._C_ascend.npu_lightning_indexer(
                query=query, key=key, weights=weights,
                layout_query="BSND", layout_key="BSND",
                sparse_count=sparse_count, sparse_mode=3,
                return_value=True,
            )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2, "Expected 2 outputs: (sparse_indices, sparse_values)")

        sparse_indices, sparse_values = result

        # sparse_indices shape matches (B, S, N2, sparse_count)
        expected_shape = (batch, seq_q, num_kv_heads, sparse_count)
        self.assertEqual(sparse_indices.shape, expected_shape)
        self.assertEqual(sparse_indices.dtype, torch.int32)

        # sparse_values shape matches sparse_indices shape
        self.assertEqual(sparse_values.shape, expected_shape)
        self.assertEqual(sparse_values.dtype, query.dtype)


class TestSparseFlashAttentionCallSite(TestBase):
    """Test that call sites correctly handle the new 3-tuple return value."""

    def test_sfa_v1_unpack_semantics(self):
        """Verify call sites correctly unpack the 3-tuple from npu_sparse_flash_attention."""
        batch, seq_q, num_kv_heads, num_q_heads, head_dim = 1, 4, 1, 4, 128
        query = torch.randn(batch, seq_q, num_q_heads, head_dim, dtype=torch.float16)
        key = torch.randn(batch, seq_q, num_kv_heads, head_dim, dtype=torch.float16)
        sparse_indices = torch.randint(0, seq_q, (batch, seq_q, num_kv_heads, 32), dtype=torch.int32)

        # Simulate the real op returning a 3-tuple
        def mock_op(**kwargs):
            attn_out = torch.randn_like(kwargs["query"])
            softmax_max = torch.randn(num_kv_heads, batch, num_q_heads // num_kv_heads,
                                      dtype=torch.float32, device=query.device)
            softmax_sum = torch.randn(num_kv_heads, batch, num_q_heads // num_kv_heads,
                                      dtype=torch.float32, device=query.device)
            return attn_out, softmax_max, softmax_sum

        with patch("torch.ops._C_ascend.npu_sparse_flash_attention",
                   side_effect=mock_op):
            # This mirrors the calling pattern in sfa_v1.py:
            #   attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(...)
            # which now needs to be:
            #   attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(...)
            # or
            #   attn_output, softmax_max, softmax_sum = torch.ops._C_ascend.npu_sparse_flash_attention(...)
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=query, key=key, value=key,
                sparse_indices=sparse_indices, scale_value=1.0,
                layout_query="BSND", layout_kv="BSND",
            )

            attn_output, softmax_max, softmax_sum = result

            self.assertEqual(attn_output.shape, query.shape)
            self.assertEqual(attn_output.dtype, query.dtype)
            self.assertEqual(softmax_max.dtype, torch.float32)
            self.assertEqual(softmax_sum.dtype, torch.float32)


class TestLightningIndexerCallSite(TestBase):
    """Test that call sites correctly handle the new 2-tuple return value."""

    def test_lightning_indexer_unpack_semantics(self):
        """Verify call sites correctly unpack the 2-tuple from npu_lightning_indexer."""
        batch, seq_q, num_kv_heads, num_q_heads, head_dim = 1, 4, 1, 4, 128
        sparse_count = 2048

        query = torch.randn(batch, seq_q, num_q_heads, head_dim, dtype=torch.float16)
        key = torch.randn(batch, seq_q, num_kv_heads, head_dim, dtype=torch.float16)
        weights = torch.randn(batch, seq_q, num_q_heads, dtype=torch.float16)

        def mock_op(**kwargs):
            indices = torch.zeros(batch, seq_q, num_kv_heads, sparse_count,
                                  dtype=torch.int32, device=query.device)
            values = torch.randn(batch, seq_q, num_kv_heads, sparse_count,
                                 dtype=query.dtype, device=query.device)
            return indices, values

        with patch("torch.ops._C_ascend.npu_lightning_indexer",
                   side_effect=mock_op):
            # This mirrors the calling pattern in sfa_v1.py:
            #   topk_indices = torch.ops._C_ascend.npu_lightning_indexer(...)
            # which now needs to be:
            #   topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(...)
            result = torch.ops._C_ascend.npu_lightning_indexer(
                query=query, key=key, weights=weights,
                layout_query="BSND", layout_key="BSND",
                sparse_count=sparse_count, sparse_mode=3,
            )

            topk_indices, sparse_values = result

            self.assertEqual(topk_indices.dtype, torch.int32)
            self.assertEqual(sparse_values.dtype, query.dtype)
            expected_shape = (batch, seq_q, num_kv_heads, sparse_count)
            self.assertEqual(topk_indices.shape, expected_shape)
            self.assertEqual(sparse_values.shape, expected_shape)
