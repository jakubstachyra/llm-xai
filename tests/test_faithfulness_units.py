import unittest
import numpy as np
import transformers
from src.faithfulness.units import (
    WordUnit,
    extract_word_units,
    aggregate_word_attributions,
    rank_word_units,
)


class TestFaithfulnessUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use Phi-2 tokenizer for realistic BPE subtoken testing
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )

    def test_word_unit_extraction_and_character_slices(self):
        text = "A masterfully-crafted, deeply moving film... it's amazing!"
        units = extract_word_units(text, self.tokenizer)

        assert len(units) > 0, "Expected non-empty units"
        
        # Verify that each unit's text matches the exact character slice from text
        for unit in units:
            assert text[unit.start_char:unit.end_char] == unit.text, (
                f"Mismatch: slice '{text[unit.start_char:unit.end_char]}' != text '{unit.text}'"
            )

        # Check that 'masterfully' was properly grouped from multiple subtokens
        masterfully_unit = next((u for u in units if u.text == "masterfully"), None)
        assert masterfully_unit is not None, "Expected 'masterfully' unit"
        assert len(masterfully_unit.tokens) == 2, f"Expected 2 subtokens, got {masterfully_unit.tokens}"
        assert len(masterfully_unit.token_indices) == 2

        # Check first token 'A'
        first_unit = units[0]
        assert first_unit.text == "A"
        assert first_unit.start_char == 0
        assert first_unit.end_char == 1

        # Check last token '!'
        last_unit = units[-1]
        assert last_unit.text == "!"
        assert last_unit.end_char == len(text)

    def test_signed_aggregation_methods(self):
        units = [
            WordUnit(index=0, text="good", start_char=0, end_char=4, token_indices=[2, 3]),
            WordUnit(index=1, text="bad", start_char=5, end_char=8, token_indices=[4]),
        ]
        
        # Subtoken scores: prompt tokens 0..5
        # Index 2: +0.3, Index 3: -0.1 -> sum = +0.2, max = +0.3
        # Index 4: -0.5
        token_attributions = np.array([0.0, 0.0, 0.3, -0.1, -0.5, 0.0])

        # Test signed sum
        aggregate_word_attributions(units, token_attributions, method="signed_sum")
        assert np.isclose(units[0].score, 0.2), f"Expected 0.2, got {units[0].score}"
        assert np.isclose(units[1].score, -0.5), f"Expected -0.5, got {units[1].score}"

        # Test signed max
        aggregate_word_attributions(units, token_attributions, method="signed_max")
        assert np.isclose(units[0].score, 0.3), f"Expected 0.3, got {units[0].score}"
        assert np.isclose(units[1].score, -0.5), f"Expected -0.5, got {units[1].score}"

        # Test mean
        aggregate_word_attributions(units, token_attributions, method="mean")
        assert np.isclose(units[0].score, 0.1), f"Expected 0.1, got {units[0].score}"
        assert np.isclose(units[1].score, -0.5), f"Expected -0.5, got {units[1].score}"

    def test_deterministic_rankings_and_tie_breaking(self):
        units = [
            WordUnit(index=0, text="word0", start_char=0, end_char=5, score=0.5),
            WordUnit(index=1, text="word1", start_char=6, end_char=11, score=-0.8),
            WordUnit(index=2, text="word2", start_char=12, end_char=17, score=0.5), # Tie with word0
            WordUnit(index=3, text="word3", start_char=18, end_char=23, score=0.01),
        ]

        # Top signed: highest score first, ties broken by ascending index
        top_ranked = rank_word_units(units, ranking_mode="top_signed")
        assert [u.index for u in top_ranked] == [0, 2, 3, 1], f"Unexpected top ranking: {[u.index for u in top_ranked]}"

        # Bottom signed: most negative score first
        bottom_ranked = rank_word_units(units, ranking_mode="bottom_signed")
        assert [u.index for u in bottom_ranked] == [1, 3, 0, 2], f"Unexpected bottom ranking: {[u.index for u in bottom_ranked]}"

        # Near zero: smallest absolute score first
        near_zero_ranked = rank_word_units(units, ranking_mode="near_zero")
        assert [u.index for u in near_zero_ranked] == [3, 0, 2, 1], f"Unexpected near zero ranking: {[u.index for u in near_zero_ranked]}"

        # Random mode: reproducible with seed
        rand1 = rank_word_units(units, ranking_mode="random", seed=42)
        rand2 = rank_word_units(units, ranking_mode="random", seed=42)
        assert [u.index for u in rand1] == [u.index for u in rand2], "Random ranking must be deterministic with seed"

    def test_word_unit_extraction_qwen_tokenizer(self):
        qwen_tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        text = "A masterfully-crafted, deeply moving film... it's amazing!"
        units = extract_word_units(text, qwen_tokenizer)

        assert len(units) > 0
        for unit in units:
            assert text[unit.start_char:unit.end_char] == unit.text, (
                f"Mismatch: slice '{text[unit.start_char:unit.end_char]}' != text '{unit.text}'"
            )

        masterfully_unit = next((u for u in units if u.text == "masterfully"), None)
        assert masterfully_unit is not None
        assert len(masterfully_unit.token_indices) == 2, f"Expected 2 subtokens for masterfully, got {masterfully_unit.tokens}"

        # Check grouping on multi-subtoken words
        text2 = "It was completely unwatchable."
        units2 = extract_word_units(text2, qwen_tokenizer)
        unwatchable_unit = next((u for u in units2 if u.text == "unwatchable"), None)
        assert unwatchable_unit is not None
        assert len(unwatchable_unit.token_indices) == 3, f"Expected 3 subtokens for unwatchable, got {unwatchable_unit.tokens}"

