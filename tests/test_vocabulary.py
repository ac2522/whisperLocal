"""Tests for engine.vocabulary — prompt building + fuzzy post-substitution."""

import pytest

from engine.vocabulary import apply_post_substitution, build_whisper_prompt


class TestBuildWhisperPrompt:
    def test_empty_list_returns_none(self):
        assert build_whisper_prompt([]) is None

    def test_none_input_returns_none(self):
        assert build_whisper_prompt(None) is None

    def test_single_entry(self):
        assert build_whisper_prompt(["Avrillo"]) == "Avrillo"

    def test_multiple_entries_comma_joined(self):
        result = build_whisper_prompt(["Avrillo", "conveyancing", "SDLT"])
        assert result == "Avrillo, conveyancing, SDLT"

    def test_strips_and_drops_empty_entries(self):
        result = build_whisper_prompt(["  Avrillo  ", "", "SDLT"])
        assert result == "Avrillo, SDLT"

    def test_truncates_at_word_boundary(self):
        long_entries = [f"word{i}" for i in range(60)]  # "word0, word1, ..., word59"
        result = build_whisper_prompt(long_entries, max_chars=50)
        assert len(result) <= 50
        # Must end on a word, not a partial one or a comma
        assert not result.endswith(",")
        assert not result.endswith(", ")
        # All included words must be complete
        for token in result.split(", "):
            assert token in long_entries

    def test_returns_none_if_first_entry_exceeds_budget(self):
        assert build_whisper_prompt(["supercalifragilistic"], max_chars=5) is None


class TestApplyPostSubstitution:
    def test_no_vocab_returns_unchanged(self):
        assert apply_post_substitution("hello world", []) == "hello world"
        assert apply_post_substitution("hello world", None) == "hello world"

    def test_exact_case_insensitive_match_replaced(self):
        # Lowercase source, title-case vocab with no interior uppercase:
        # source case wins → stays lowercase.
        assert apply_post_substitution("visit avrillo today", ["Avrillo"]) == "visit avrillo today"
        # Exact vocab match passes through unchanged.
        assert apply_post_substitution("Avrillo", ["Avrillo"]) == "Avrillo"

    def test_fuzzy_match_above_cutoff(self):
        # "Avrilo" (missing an 'l') should match "Avrillo"
        result = apply_post_substitution("Visit Avrilo today", ["Avrillo"])
        assert "Avrillo" in result
        assert "Avrilo" not in result

    def test_below_cutoff_left_alone(self):
        # "hello" and "Avrillo" are too dissimilar; no substitution
        result = apply_post_substitution("hello there", ["Avrillo"])
        assert result == "hello there"

    def test_case_preservation_lower(self):
        result = apply_post_substitution("avrilo", ["Avrillo"])
        assert result == "avrillo"

    def test_case_preservation_title(self):
        result = apply_post_substitution("Avrilo", ["Avrillo"])
        assert result == "Avrillo"

    def test_case_preservation_upper(self):
        result = apply_post_substitution("AVRILO", ["Avrillo"])
        assert result == "AVRILLO"

    def test_punctuation_preserved(self):
        result = apply_post_substitution("We use Avrilo.", ["Avrillo"])
        assert result == "We use Avrillo."

    def test_multiple_substitutions_in_one_string(self):
        result = apply_post_substitution(
            "avrilo handles SDLTs on sdlt matters",
            ["Avrillo", "SDLT"],
        )
        assert "avrillo" in result.lower()
        assert "SDLT" in result

    def test_short_common_words_not_false_positive(self):
        # "the" shouldn't be turned into "they" or any other short vocab entry
        result = apply_post_substitution("the quick brown fox", ["they"])
        assert result == "the quick brown fox"

    def test_word_boundaries_respected(self):
        # "rill" inside "drilling" should not become "drillingll" or get replaced
        result = apply_post_substitution("drilling holes", ["rill"])
        assert result == "drilling holes"

    def test_cutoff_threshold_controls_aggressiveness(self):
        # At default cutoff 0.85, "Evrelo" vs "Avrillo" is too far apart
        result = apply_post_substitution("visit Evrelo", ["Avrillo"])
        assert "Evrelo" in result  # unchanged

    def test_interior_uppercase_preserved_for_lower_source(self):
        # "iphone" in transcript, "iPhone" in vocab — brand casing wins.
        assert apply_post_substitution("my iphone", ["iPhone"]) == "my iPhone"

    def test_interior_uppercase_preserved_for_title_source(self):
        # "Iphone" in transcript, "iPhone" in vocab — still brand casing.
        assert apply_post_substitution("Iphone", ["iPhone"]) == "iPhone"

    def test_interior_uppercase_promoted_to_upper_when_source_shouts(self):
        # "IPHONE" in transcript → "IPHONE" (all caps wins).
        assert apply_post_substitution("IPHONE", ["iPhone"]) == "IPHONE"

    def test_possessive_apostrophe_tokens_substitute_word_part(self):
        # Avrillo's should match Avrillo and leave the 's intact.
        assert apply_post_substitution("Avrilo's office", ["Avrillo"]) == "Avrillo's office"

    def test_mixed_case_brand_like_MacDonald(self):
        assert apply_post_substitution("see macdonald tomorrow", ["MacDonald"]) == "see MacDonald tomorrow"
