"""
Unit tests for top_logprob optimization in custom_renderer.py

This test suite validates the optimization that uses torch.topk instead of sort 
for better computational efficiency when generating log probabilities.
"""

import unittest
from unittest.mock import MagicMock, Mock

import torch

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionTokenLogprob,
    ChatMessage,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory import RendererParams
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    StreamStatus,
    StreamStatusSync,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


class TestTopLogprobOptimization(unittest.TestCase):
    """Test suite for validating the topk optimization in log probability generation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = 0
        # Mock decode function to return token representation

        def mock_decode(ids):
            if isinstance(ids, list):
                return f"token_{ids[0]}"
            return f"token_{ids}"

        self.tokenizer.decode = Mock(side_effect=mock_decode)

        # Create renderer with mock parameters
        render_params = RendererParams(
            model_type="test_model",
            max_seq_len=1024,
            eos_token_id=0,
            stop_word_ids_list=[],
        )
        self.renderer = CustomChatRenderer(self.tokenizer, render_params)

    def test_generate_log_probs_basic(self):
        """Test basic log probability generation with topk"""
        # Create test data
        vocab_size = 100
        all_probs = torch.zeros(vocab_size)
        # Set probabilities for top 5 tokens
        top_indices = [10, 20, 30, 40, 50]
        top_probs = [0.5, 0.2, 0.15, 0.1, 0.05]
        for idx, prob in zip(top_indices, top_probs):
            all_probs[idx] = prob

        # Create output
        selected_id = 10  # token with highest probability
        output_ids = torch.tensor([[selected_id]])
        output = GenerateOutput(
            hidden_states=None,
            output_ids=output_ids,
            finished=torch.tensor([False]),
            aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
            loss=None,
            logits=None,
            all_probs=all_probs.unsqueeze(0),
        )

        # Create request with logprobs enabled
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=3,
        )

        # Create status
        status = StreamStatus(request)

        # Test async version
        import asyncio
        result = asyncio.run(self.renderer._generate_log_probs(status, output))  # type: ignore

        # Validate results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ChatCompletionTokenLogprob)
        assert result is not None  # Type narrowing
        self.assertEqual(len(result.top_logprobs), 3)

        # Verify that top logprobs are in descending order
        logprobs_values = [lp.logprob for lp in result.top_logprobs]
        self.assertEqual(logprobs_values, sorted(logprobs_values, reverse=True))

        # Verify the selected token has the highest probability
        self.assertAlmostEqual(result.logprob, torch.log(torch.tensor(0.5)).item(), places=5)

    def test_generate_log_probs_sync_basic(self):
        """Test basic log probability generation with topk (sync version)"""
        # Create test data
        vocab_size = 100
        all_probs = torch.zeros(vocab_size)
        # Set probabilities for top 5 tokens
        top_indices = [15, 25, 35, 45, 55]
        top_probs = [0.4, 0.25, 0.2, 0.1, 0.05]
        for idx, prob in zip(top_indices, top_probs):
            all_probs[idx] = prob

        # Create output
        selected_id = 15  # token with highest probability
        output_ids = torch.tensor([[selected_id]])

        # Create request with logprobs enabled
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=4,
        )

        # Create status
        status = StreamStatusSync(request)

        # Test sync version
        result = self.renderer._generate_log_probs_sync(status, all_probs.unsqueeze(0), output_ids)  # type: ignore

        # Validate results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ChatCompletionTokenLogprob)
        assert result is not None  # Type narrowing
        self.assertEqual(len(result.top_logprobs), 4)

        # Verify that top logprobs are in descending order
        logprobs_values = [lp.logprob for lp in result.top_logprobs]
        self.assertEqual(logprobs_values, sorted(logprobs_values, reverse=True))

    def test_topk_with_fewer_nonzero_than_requested(self):
        """Test when number of non-zero probabilities is less than requested top_k"""
        # Create test data with only 2 non-zero probabilities
        vocab_size = 100
        all_probs = torch.zeros(vocab_size)
        all_probs[10] = 0.7
        all_probs[20] = 0.3

        selected_id = 10
        output_ids = torch.tensor([[selected_id]])
        output = GenerateOutput(
            hidden_states=None,
            output_ids=output_ids,
            finished=False,
            aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
            loss=None,
            logits=None,
            all_probs=all_probs.unsqueeze(0),
        )

        # Request more top_logprobs than available
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=5,
        )

        status = StreamStatus(request)

        import asyncio
        result = asyncio.run(self.renderer._generate_log_probs(status, output))  # type: ignore

        # Should only return 2 logprobs (number of non-zero elements)
        assert result is not None  # Type narrowing
        self.assertEqual(len(result.top_logprobs), 2)

    def test_topk_efficiency_comparison(self):
        """Test that topk is more efficient than full sort for large vocabularies"""
        import time

        # Simulate a large vocabulary
        vocab_size = 50000
        all_probs = torch.rand(vocab_size)
        all_probs = all_probs / all_probs.sum()  # Normalize

        top_k = 5

        # Method 1: Using topk (optimized)
        probs_topk = None
        tokens_topk = None
        start_time = time.time()
        for _ in range(100):
            probs_topk, tokens_topk = all_probs.topk(top_k, dim=-1, largest=True, sorted=True)
        topk_time = time.time() - start_time

        # Method 2: Using sort (old approach)
        probs_sort = None
        tokens_sort = None
        start_time = time.time()
        for _ in range(100):
            probs_sort, tokens_sort = all_probs.sort(descending=True)
            probs_sort = probs_sort[:top_k]
            tokens_sort = tokens_sort[:top_k]
        sort_time = time.time() - start_time

        # topk should be faster than sort
        print(f"topk time: {topk_time:.4f}s, sort time: {sort_time:.4f}s")
        print(f"Speedup: {sort_time / topk_time:.2f}x")
        self.assertLess(topk_time, sort_time, "topk should be faster than full sort")

        # Verify results are the same
        assert probs_topk is not None and probs_sort is not None
        assert tokens_topk is not None and tokens_sort is not None
        self.assertTrue(torch.allclose(probs_topk, probs_sort))
        self.assertTrue(torch.equal(tokens_topk, tokens_sort))

    def test_topk_correctness_against_sort(self):
        """Verify that topk produces the same results as sort for top-k elements"""
        vocab_size = 1000
        all_probs = torch.rand(vocab_size)
        top_k = 10

        # Using topk
        probs_topk, tokens_topk = all_probs.topk(top_k, dim=-1, largest=True, sorted=True)

        # Using sort
        probs_sort, tokens_sort = all_probs.sort(descending=True)
        probs_sort = probs_sort[:top_k]
        tokens_sort = tokens_sort[:top_k]

        # Results should match
        self.assertTrue(torch.allclose(probs_topk, probs_sort, rtol=1e-5))
        self.assertTrue(torch.equal(tokens_topk, tokens_sort))

    def test_logprobs_disabled(self):
        """Test that None is returned when logprobs is disabled"""
        output = GenerateOutput(
            hidden_states=None,
            output_ids=torch.tensor([[10]]),
            finished=False,
            aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
            loss=None,
            logits=None,
            all_probs=torch.rand(100).unsqueeze(0),
        )

        # Request without logprobs
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=False,
        )

        status = StreamStatus(request)

        import asyncio
        result = asyncio.run(self.renderer._generate_log_probs(status, output))  # type: ignore

        self.assertIsNone(result)

    def test_all_probs_none_raises_exception(self):
        """Test that exception is raised when all_probs is None but logprobs is True"""
        output = GenerateOutput(
            hidden_states=None,
            output_ids=torch.tensor([[10]]),
            finished=False,
            aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
            loss=None,
            logits=None,
            all_probs=None,  # None value
        )

        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=3,
        )

        status = StreamStatus(request)

        import asyncio
        with self.assertRaises(Exception) as context:
            asyncio.run(self.renderer._generate_log_probs(status, output))  # type: ignore

        self.assertIn("all_probs is None", str(context.exception))

    def test_default_top_logprobs_value(self):
        """Test default top_logprobs value when not specified"""
        vocab_size = 100
        all_probs = torch.rand(vocab_size)
        all_probs = all_probs / all_probs.sum()

        output = GenerateOutput(
            hidden_states=None,
            output_ids=torch.tensor([[10]]),
            finished=False,
            aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
            loss=None,
            logits=None,
            all_probs=all_probs.unsqueeze(0),
        )

        # Request with logprobs but no top_logprobs specified
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            # top_logprobs not specified, should default to 1
        )

        status = StreamStatus(request)

        import asyncio
        result = asyncio.run(self.renderer._generate_log_probs(status, output))  # type: ignore

        # Should return 1 logprob by default
        assert result is not None  # Type narrowing
        self.assertEqual(len(result.top_logprobs), 1)

    def test_edge_case_single_nonzero_probability(self):
        """Test edge case with only one non-zero probability"""
        vocab_size = 100
        all_probs = torch.zeros(vocab_size)
        all_probs[42] = 1.0  # Only one token has probability

        output = GenerateOutput(
            hidden_states=None,
            output_ids=torch.tensor([[42]]),
            finished=False,
            aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
            loss=None,
            logits=None,
            all_probs=all_probs.unsqueeze(0),
        )

        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=5,
        )

        status = StreamStatus(request)

        import asyncio
        result = asyncio.run(self.renderer._generate_log_probs(status, output))  # type: ignore

        # Should only return 1 logprob
        assert result is not None  # Type narrowing
        self.assertEqual(len(result.top_logprobs), 1)
        self.assertAlmostEqual(result.logprob, 0.0, places=5)  # log(1.0) = 0


if __name__ == "__main__":
    unittest.main()
