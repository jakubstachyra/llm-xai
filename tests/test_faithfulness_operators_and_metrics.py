import pytest
import torch
import numpy as np
from src.faithfulness import (
    WordUnit,
    apply_word_deletion,
    apply_word_retention,
    apply_embedding_mask,
    evaluate_faithfulness_curves,
    calculate_aopc,
    compute_normalized_gap,
)


class TestFaithfulnessOperatorsAndMetrics:
    def test_discrete_word_deletion_and_retention(self):
        text = "The movie was extraordinarily fantastic and brilliant!"
        # Manually create mock WordUnits with exact character offsets
        # "extraordinarily" is at chars 14:29
        # "fantastic" is at chars 30:39
        unit_extra = WordUnit(index=0, text="extraordinarily", start_char=14, end_char=29, score=0.8)
        unit_fantastic = WordUnit(index=1, text="fantastic", start_char=30, end_char=39, score=0.9)
        unit_brilliant = WordUnit(index=2, text="brilliant", start_char=44, end_char=53, score=0.7)
        all_units = [unit_extra, unit_fantastic, unit_brilliant]

        # Deletion of Top-1 ('fantastic')
        deleted_text = apply_word_deletion(text, [unit_fantastic])
        assert "fantastic" not in deleted_text
        assert "extraordinarily" in deleted_text
        assert "brilliant" in deleted_text
        assert deleted_text == "The movie was extraordinarily and brilliant!"

        # Deletion of multiple units
        deleted_two = apply_word_deletion(text, [unit_extra, unit_fantastic])
        assert "extraordinarily" not in deleted_two
        assert "fantastic" not in deleted_two
        assert deleted_two == "The movie was and brilliant!"

        # Retention of only 'fantastic'
        retained = apply_word_retention(text, all_units, [unit_fantastic])
        assert "fantastic" in retained
        assert "extraordinarily" not in retained
        assert "brilliant" not in retained

    def test_embedding_mask_operator(self):
        # Batch size 1, sequence length 6, embedding dim 4
        embeds = torch.ones((1, 6, 4), dtype=torch.float32)
        unit = WordUnit(index=0, text="test", start_char=0, end_char=4, token_indices=[2, 3])

        masked = apply_embedding_mask(embeds, [unit], mask_value="zero")
        
        # Indices 2 and 3 should be exactly zero
        assert torch.all(masked[:, 2, :] == 0.0)
        assert torch.all(masked[:, 3, :] == 0.0)
        
        # Indices 0, 1, 4, 5 should remain untouched (all ones)
        assert torch.all(masked[:, 0, :] == 1.0)
        assert torch.all(masked[:, 1, :] == 1.0)
        assert torch.all(masked[:, 4, :] == 1.0)
        assert torch.all(masked[:, 5, :] == 1.0)

    def test_metrics_aopc_and_normalized_gap(self):
        k_values = [1, 2, 3]
        c_prob = [0.10, 0.20, 0.30] # mean = 0.20
        c_log = [0.20, 0.40, 0.60]  # mean = 0.40
        u_prob = [0.05, 0.05, 0.05] # mean = 0.05
        u_log = [0.10, 0.10, 0.10]  # mean = 0.10
        random_aopc = 0.08

        res = evaluate_faithfulness_curves(
            k_values=k_values,
            c_prob=c_prob,
            c_log=c_log,
            u_prob=u_prob,
            u_log=u_log,
            random_aopc_c_prob=random_aopc
        )

        assert np.isclose(res.aopc_comprehensiveness_prob, 0.20)
        assert np.isclose(res.aopc_comprehensiveness_log, 0.40)
        assert np.isclose(res.aopc_sufficiency_prob, 0.05)
        assert np.isclose(res.harmonization_h, 0.20 - 0.05)

        # G metric: (0.20 - 0.08) / 0.08 = 0.12 / 0.08 = 1.50 (+150% gain over random)
        expected_g = (0.20 - 0.08) / (0.08 + 1e-6)
        assert np.isclose(res.normalized_gap_g, expected_g, atol=1e-3)
