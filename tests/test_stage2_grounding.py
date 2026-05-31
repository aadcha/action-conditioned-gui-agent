"""Pure-logic tests for the Stage 2 model.

We DON'T instantiate Qwen2-VL here (it would download 4 GB of weights on
import). Instead we directly test `_embed_with_action`'s slot-replacement
logic via a minimal stand-in.
"""

import pytest
import torch
import torch.nn as nn

from src.models.stage2_grounding import ACTION_SLOT_TOKEN


class _MiniStage2(nn.Module):
    """Replica of Stage2ConditionedGrounding._embed_with_action, isolated so
    we can test the swap logic without loading the real VLM.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_actions: int, slot_token_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.action_embeddings = nn.Embedding(num_actions, hidden_dim)
        self.action_slot_token_id = slot_token_id

    def _embed_with_action(self, input_ids, action_type_id):
        inputs_embeds = self.embed(input_ids)
        is_slot = input_ids == self.action_slot_token_id
        slot_count = is_slot.sum(dim=1)
        if (slot_count != 1).any():
            bad = (slot_count != 1).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"rows {bad} have wrong slot count {slot_count[bad].tolist()}")
        slot_positions = is_slot.nonzero(as_tuple=False)
        slot_positions = slot_positions[slot_positions[:, 0].argsort()]
        batch_idx = slot_positions[:, 0]
        time_idx = slot_positions[:, 1]
        action_embeds = self.action_embeddings(action_type_id[batch_idx]).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_idx, time_idx] = action_embeds
        return inputs_embeds


def test_swap_replaces_slot_embedding_only():
    torch.manual_seed(0)
    m = _MiniStage2(vocab_size=100, hidden_dim=8, num_actions=8, slot_token_id=99)
    # Two rows: slot at position 2 in row 0, position 4 in row 1
    input_ids = torch.tensor([
        [1, 2, 99, 3, 4, 5],
        [10, 11, 12, 13, 99, 14],
    ])
    action_type_id = torch.tensor([3, 7])

    original = m.embed(input_ids)
    out = m._embed_with_action(input_ids, action_type_id)

    # Non-slot positions are byte-identical
    mask = input_ids != 99
    assert torch.equal(out[mask], original[mask])

    # Slot positions equal the action embedding for that row's action id
    assert torch.equal(out[0, 2], m.action_embeddings(torch.tensor(3)))
    assert torch.equal(out[1, 4], m.action_embeddings(torch.tensor(7)))


def test_swap_errors_on_missing_slot():
    m = _MiniStage2(vocab_size=100, hidden_dim=4, num_actions=8, slot_token_id=99)
    input_ids = torch.tensor([[1, 2, 3, 4]])  # no slot token
    action_type_id = torch.tensor([0])
    with pytest.raises(ValueError):
        m._embed_with_action(input_ids, action_type_id)


def test_swap_errors_on_multiple_slots():
    m = _MiniStage2(vocab_size=100, hidden_dim=4, num_actions=8, slot_token_id=99)
    input_ids = torch.tensor([[99, 1, 99]])
    action_type_id = torch.tensor([0])
    with pytest.raises(ValueError):
        m._embed_with_action(input_ids, action_type_id)


def test_action_slot_token_constant_is_unique():
    # It's a special token — must contain something tokenizers won't fragment
    assert ACTION_SLOT_TOKEN == "<|action_slot|>"
