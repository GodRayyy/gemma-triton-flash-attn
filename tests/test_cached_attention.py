"""Regression tests for the HF adapter's cached decoding (no model download).

Run after an editable install: python -m unittest discover -s tests -p test_cached_attention.py -v
CUDA cases exercise Gemma4-E4B's Hq=8/Hkv=2, D=256/512 shapes.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gemma_triton_flash_attn.attention import flash_attn_gqa_train
from gemma_triton_flash_attn.hf_integration import triton_gqa_attention


def reference(q, k, v, mask, scale):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = (q.double() @ k.double().transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    probs = scores.softmax(-1).nan_to_num()
    return (probs @ v.double()).to(q.dtype).transpose(1, 2)


def causal_mask(nq, nk, window, device):
    # Construct full self-attention, then select the suffix queries.
    mask = torch.ones(nk, nk, dtype=torch.bool, device=device).tril()
    if window:
        mask = mask.triu(1 - window)
    return mask[-nq:]


class CachedAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_masks_scales_and_gradients(self):
        for kind in ('none', 'padding', 'bool4d', 'additive4d'):
            for nq in (1, 3):
                for window in (0, 4):
                    with self.subTest(kind=kind, nq=nq, window=window):
                        q = torch.randn(2, 8, nq, 8, dtype=torch.float64, requires_grad=True)
                        k = torch.randn(2, 2, 9, 8, dtype=torch.float64, requires_grad=True)
                        v = torch.randn_like(k, requires_grad=True)
                        expected_mask = causal_mask(nq, 9, window, q.device)[None, None].expand(2, 1, -1, -1).clone()
                        supplied_mask = None
                        if kind != 'none':
                            padding = torch.ones(2, 9, dtype=torch.bool)
                            padding[0, :2] = False
                            padding[1, :3] = False
                            expected_mask &= padding[:, None, None, :]
                            supplied_mask = padding if kind == 'padding' else expected_mask
                            if kind == 'additive4d':
                                supplied_mask = torch.zeros_like(expected_mask, dtype=q.dtype).masked_fill(
                                    ~expected_mask, -torch.inf)
                        module = SimpleNamespace(head_dim=8, is_causal=True)
                        actual, weights = triton_gqa_attention(
                            module, q, k, v, supplied_mask, scaling=1.0, sliding_window=window)
                        expected = reference(q, k, v, expected_mask, 1.0)
                        self.assertIsNone(weights)
                        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                        grad = torch.randn_like(actual)
                        actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
                        expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)

    def test_decode_reads_cached_tokens(self):
        q = torch.zeros(1, 8, 1, 8)
        k = torch.zeros(1, 2, 64, 8)
        v = torch.arange(64, dtype=torch.float32)[None, None, :, None].expand_as(k)
        module = SimpleNamespace(head_dim=8, is_causal=True)
        for window, expected in ((0, 31.5), (4, 61.5)):
            out, _ = triton_gqa_attention(module, q, k, v, None, sliding_window=window)
            torch.testing.assert_close(out, torch.full_like(out, expected))

    def test_static_cache_mask_is_authoritative(self):
        q = torch.zeros(1, 8, 1, 8)
        k = torch.zeros(1, 2, 16, 8)
        v = torch.arange(16, dtype=torch.float32)[None, None, :, None].expand_as(k)
        mask = torch.zeros(1, 1, 1, 16, dtype=torch.bool)
        mask[..., 2:5] = True
        out, _ = triton_gqa_attention(SimpleNamespace(head_dim=8, is_causal=True),
                                     q, k, v, mask, sliding_window=3)
        torch.testing.assert_close(out, torch.full_like(out, 3.0))
        out, _ = triton_gqa_attention(SimpleNamespace(head_dim=8, is_causal=True),
                                     q, k, v, torch.zeros_like(mask))
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_short_prefill_and_noncausal(self):
        for causal in (True, False):
            for nq, nk in ((1, 1), (7, 7), (3, 9)):
                q = torch.randn(1, 4, nq, 8, dtype=torch.float64)
                k = torch.randn(1, 2, nk, 8, dtype=torch.float64)
                v = torch.randn_like(k)
                module = SimpleNamespace(head_dim=8, is_causal=causal)
                actual, _ = triton_gqa_attention(module, q, k, v, None)
                mask = causal_mask(nq, nk, 0, q.device) if causal else None
                torch.testing.assert_close(actual, reference(q, k, v, mask, 8 ** -0.5))

    def test_square_training_still_uses_triton(self):
        q = torch.randn(1, 8, 32, 64)
        k = torch.randn(1, 2, 32, 64)
        with patch('gemma_triton_flash_attn.hf_integration.flash_attn_gqa_train', return_value=q) as kernel:
            with patch('gemma_triton_flash_attn.hf_integration.torch.cuda.device'):
                out, _ = triton_gqa_attention(SimpleNamespace(head_dim=64, is_causal=True), q, k, k, None)
        kernel.assert_called_once()
        torch.testing.assert_close(out, q.transpose(1, 2))

    def test_raw_training_kernel_rejects_rectangular_inputs(self):
        with self.assertRaisesRegex(ValueError, 'equal Q/K/V'):
            flash_attn_gqa_train(torch.empty(1, 8, 1, 8), torch.empty(1, 2, 64, 8),
                                 torch.empty(1, 2, 64, 8), causal=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_e4b_cuda_shapes(self):
        worst_relative_l2 = 0.0
        worst_scaled_max = 0.0
        for dtype in (torch.bfloat16, torch.float16):
            for dim, window in ((256, 512), (512, 0)):
                for nq, nk in ((1, 64), (16, 64), (1, 600), (16, 600)):
                    with self.subTest(dtype=dtype, dim=dim, nq=nq, nk=nk):
                        q = torch.randn(1, 8, nq, dim, device='cuda', dtype=dtype) * 0.25
                        k = torch.randn(1, 2, nk, dim, device='cuda', dtype=dtype) * 0.25
                        v = torch.randn_like(k)
                        q.requires_grad_()
                        k.requires_grad_()
                        v.requires_grad_()
                        actual, _ = triton_gqa_attention(SimpleNamespace(head_dim=dim, is_causal=True),
                                                       q, k, v, None, scaling=1.0, sliding_window=window)
                        mask = causal_mask(nq, nk, window, q.device)
                        expected = reference(q, k, v, mask, 1.0)
                        torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.04)
                        grad = torch.randn_like(actual)
                        actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
                        expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            # GQA sums contributions across heads. BF16 rounding
                            # can dominate individual near-zero entries after
                            # cancellation; compare tensor-relative errors.
                            error = actual_grad.float() - expected_grad.float()
                            relative_l2 = (error.norm() / expected_grad.float().norm()).item()
                            scaled_max = (error.abs().max() / expected_grad.float().abs().max()).item()
                            worst_relative_l2 = max(worst_relative_l2, relative_l2)
                            worst_scaled_max = max(worst_scaled_max, scaled_max)
                            self.assertLess(relative_l2, 0.015)
                            self.assertLess(scaled_max, 0.015)
        print(f'CUDA gradient worst relative L2={worst_relative_l2:.6g}, '
              f'max-error/max-reference={worst_scaled_max:.6g}')


if __name__ == '__main__':
    unittest.main()
