import re
from typing import Dict, Any, Optional

class TextPerturber:
    """
    Applies controlled, meaning-preserving text perturbations for evaluating
    explanation stability (Research Question 1 / RQ1).
    Grounding: CheckList Invariance Tests (Ribeiro et al., ACL 2020) and XAI Robustness.
    """
    
    # =========================================================================
    # 1. Surface / Orthographic Perturbations
    # =========================================================================
    @staticmethod
    def lowercase(text: str) -> str:
        """Converts text to lowercase."""
        return text.lower()

    @staticmethod
    def uppercase(text: str) -> str:
        """Converts text to uppercase (emulating shout/emphasis formatting)."""
        return text.upper()

    @staticmethod
    def add_punctuation(text: str, punct: str = ".") -> str:
        """Ensures text ends with a terminating period."""
        text_clean = text.strip()
        if text_clean.endswith((".", "!", "?")):
            return text_clean
        return f"{text_clean}{punct}"

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Removes internal and trailing punctuation marks."""
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def double_space(text: str) -> str:
        """Replaces single spaces with double spaces (tests BPE tokenizer whitespace sensitivity)."""
        words = text.strip().split()
        return "  ".join(words)

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Preserves the capitalization pattern of the original matched token."""
        if original.isupper():
            return replacement.upper()
        if original and original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement.lower()

    @staticmethod
    def typo_swap(text: str) -> str:
        """
        Simulates an innocent keyboard typo by swapping two adjacent internal
        characters in a neutral word of length >= 5 (CheckList INV standard).
        Safely ignores attached punctuation.
        """
        words = text.strip().split()
        # Find candidate words with length >= 5 that are not sentiment keywords
        sentiment_keywords = {"great", "terrible", "awful", "wonderful", "bad", "good", "boring", "worst", "best"}
        for idx, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 5 and clean_word.lower() not in sentiment_keywords:
                # Swap internal characters at positions 1 and 2
                w_list = list(clean_word)
                w_list[1], w_list[2] = w_list[2], w_list[1]
                swapped = "".join(w_list)
                words[idx] = word.replace(clean_word, swapped, 1)
                return " ".join(words)
        # Fallback: if no 5-letter word found, swap characters in the clean part of the longest word
        if words:
            longest_idx = max(range(len(words)), key=lambda i: len(re.sub(r'[^\w]', '', words[i])))
            word = words[longest_idx]
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 2:
                w_list = list(clean_word)
                w_list[0], w_list[1] = w_list[1], w_list[0]
                swapped = "".join(w_list)
                words[longest_idx] = word.replace(clean_word, swapped, 1)
        return " ".join(words)

    # =========================================================================
    # 2. Discourse / Prefix Framing Perturbations
    # =========================================================================
    @staticmethod
    def add_prefix(text: str, prefix: str = "Overall, ") -> str:
        """Adds a neutral discourse-organizing introductory prefix."""
        return f"{prefix}{text.strip()}"

    @staticmethod
    def add_prefix_opinion(text: str) -> str:
        """Adds opinion framing / hedging prefix (pragmatic discourse marker)."""
        return f"In my opinion, {text.strip()}"

    @staticmethod
    def add_prefix_in_summary(text: str) -> str:
        """Adds summary framing prefix."""
        return f"In summary, {text.strip()}"

    @staticmethod
    def add_prefix_instruction(text: str) -> str:
        """Adds prompt context instruction prefix."""
        return f"Consider this review: {text.strip()}"

    @staticmethod
    def add_prefix_frankly(text: str) -> str:
        """Adds colloquial conversational prefix."""
        return f"Frankly speaking, {text.strip()}"

    # =========================================================================
    # 3. Neutral Suffix / Domain Extension Perturbations
    # =========================================================================
    @staticmethod
    def add_neutral_suffix(text: str, suffix: str = " indeed.") -> str:
        """Appends a neutral emphatic adverb at the end."""
        text_clean = text.strip().rstrip(".")
        return f"{text_clean}{suffix}"

    @staticmethod
    def add_suffix_movie(text: str) -> str:
        """Appends domain-grounding neutral descriptor at the end."""
        text_clean = text.strip().rstrip(".")
        return f"{text_clean} for a movie."

    @staticmethod
    def add_suffix_experience(text: str) -> str:
        """Appends neutral perceptual verb phrase at the end."""
        text_clean = text.strip().rstrip(".")
        return f"{text_clean} to watch."

    # =========================================================================
    # 4. Lexical Substitution & Contractions (CheckList INV)
    # =========================================================================
    @classmethod
    def swap_domain_synonyms(cls, text: str) -> str:
        """
        Substitutes neutral domain nouns (film <-> movie, plot <-> story)
        without altering sentiment-bearing adjectives or verbs, preserving case.
        """
        replacements = [
            (r'\bfilm\b', 'movie'),
            (r'\bmovie\b', 'film'),
            (r'\bfilms\b', 'movies'),
            (r'\bmovies\b', 'films'),
            (r'\bcinema\b', 'theater'),
            (r'\bplot\b', 'storyline'),
            (r'\bdirector\b', 'filmmaker')
        ]
        text_mod = text
        for pattern, replacement in replacements:
            match = re.search(pattern, text_mod, flags=re.IGNORECASE)
            if match:
                matched_str = match.group(0)
                rep = cls._match_case(matched_str, replacement)
                # Replace only first occurrence to keep change minimal
                text_mod = re.sub(pattern, rep, text_mod, count=1, flags=re.IGNORECASE)
                return text_mod
        # Fallback if no specific movie words exist: neutral article/determiner swap
        the_match = re.search(r'\bthe\b', text_mod, flags=re.IGNORECASE)
        if the_match:
            rep = cls._match_case(the_match.group(0), 'this')
            return re.sub(r'\bthe\b', rep, text_mod, count=1, flags=re.IGNORECASE)
        return f"The film: {text_mod}"

    @classmethod
    def expand_contractions(cls, text: str) -> str:
        """
        Expands or contracts colloquial English auxiliary verbs and pronouns, preserving case.
        """
        contractions = [
            (r"\bit 's\b", "it is"),
            (r"\bit's\b", "it is"),
            (r"\bthat 's\b", "that is"),
            (r"\bthat's\b", "that is"),
            (r"\bthere 's\b", "there is"),
            (r"\bthere's\b", "there is"),
            (r"\bca n't\b", "cannot"),
            (r"\bcan't\b", "cannot"),
            (r"\bdo n't\b", "do not"),
            (r"\bdon't\b", "do not"),
            (r"\bdoes n't\b", "does not"),
            (r"\bdoesn't\b", "does not")
        ]
        text_mod = text
        for pattern, replacement in contractions:
            match = re.search(pattern, text_mod, flags=re.IGNORECASE)
            if match:
                rep = cls._match_case(match.group(0), replacement)
                text_mod = re.sub(pattern, rep, text_mod, count=1, flags=re.IGNORECASE)
                return text_mod
        # Invert: expand "is not" -> "isn't" or "it is" -> "it's"
        it_is_match = re.search(r'\bit is\b', text_mod, flags=re.IGNORECASE)
        if it_is_match:
            rep = cls._match_case(it_is_match.group(0), "it's")
            return re.sub(r'\bit is\b', rep, text_mod, count=1, flags=re.IGNORECASE)
        clean_end = text_mod.strip().rstrip(".")
        return f"{clean_end} really."

    # =========================================================================
    # Aggregation & Taxonomy Mapping
    # =========================================================================
    @classmethod
    def get_categorized_perturbations(cls, text: str) -> Dict[str, Dict[str, str]]:
        """
        Returns a complete dictionary of 16 meaning-preserving perturbations
        grouped into 4 scientific categories:
        - surface: lowercase, uppercase, remove_punct, add_period, double_space, typo_swap
        - prefix: overall, opinion, in_summary, instruction, frankly
        - suffix: indeed, movie, experience
        - lexical: domain_synonym, contractions
        """
        return {
            "surface": {
                "lowercase": cls.lowercase(text),
                "uppercase": cls.uppercase(text),
                "remove_punct": cls.remove_punctuation(text),
                "add_period": cls.add_punctuation(text, "."),
                "double_space": cls.double_space(text),
                "typo_swap": cls.typo_swap(text)
            },
            "prefix": {
                "prefix_overall": cls.add_prefix(text, "Overall, "),
                "prefix_opinion": cls.add_prefix_opinion(text),
                "prefix_in_summary": cls.add_prefix_in_summary(text),
                "prefix_instruction": cls.add_prefix_instruction(text),
                "prefix_frankly": cls.add_prefix_frankly(text)
            },
            "suffix": {
                "suffix_indeed": cls.add_neutral_suffix(text, " indeed."),
                "suffix_movie": cls.add_suffix_movie(text),
                "suffix_experience": cls.add_suffix_experience(text)
            },
            "lexical": {
                "synonym_domain": cls.swap_domain_synonyms(text),
                "contractions": cls.expand_contractions(text)
            }
        }

    @classmethod
    def get_standard_perturbations(cls, text: str) -> Dict[str, str]:
        """Flat dictionary of all perturbations."""
        all_perts = {}
        for category, perts in cls.get_categorized_perturbations(text).items():
            all_perts.update(perts)
        return all_perts

