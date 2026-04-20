"""Custom vocabulary helpers.

Two independent surfaces:
- ``build_whisper_prompt`` — build an ``initial_prompt`` string for whisper.cpp
  within a char budget, truncating at word boundaries.
- ``apply_post_substitution`` — fuzzy word-level substitution for Parakeet
  output, where no prompt-biasing mechanism exists.
"""

import difflib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z]+")


def build_whisper_prompt(
    vocab: Optional[list[str]], max_chars: int = 200
) -> Optional[str]:
    """Build an initial_prompt string for whisper.cpp from a vocab list.

    Entries are stripped and blanks removed. Multiple entries are joined with
    ", ". If the combined length exceeds ``max_chars`` the result is truncated
    at the last complete entry that fits. Returns ``None`` for empty input or
    when even the first entry already exceeds the budget.
    """
    if not vocab:
        return None

    cleaned = [e.strip() for e in vocab if isinstance(e, str) and e.strip()]
    if not cleaned:
        return None

    # Fit as many entries as possible. Stop when adding the next would exceed.
    accepted: list[str] = []
    running_len = 0
    for entry in cleaned:
        # Length if we add this entry: existing + ", " + entry
        add_len = len(entry) if not accepted else len(entry) + 2
        if running_len + add_len > max_chars:
            break
        accepted.append(entry)
        running_len += add_len

    if not accepted:
        logger.info(
            "Custom vocabulary's first entry '%s' exceeds max_chars=%d; "
            "skipping prompt biasing.", cleaned[0], max_chars,
        )
        return None

    if len(accepted) < len(cleaned):
        logger.info(
            "Custom vocabulary truncated: using %d of %d entries to fit "
            "prompt budget (%d chars).",
            len(accepted), len(cleaned), max_chars,
        )

    return ", ".join(accepted)


def apply_post_substitution(
    text: str, vocab: Optional[list[str]], cutoff: float = 0.85
) -> str:
    """Replace transcript words with near-matching vocabulary entries.

    For each word-like token in ``text``, look for a close match in ``vocab``
    using ``difflib.get_close_matches`` at the given ``cutoff``. If found,
    replace the token with the canonical vocab entry, preserving the original
    token's case pattern (all-lower / title / all-upper).
    """
    if not vocab:
        return text

    cleaned = [v.strip() for v in vocab if isinstance(v, str) and v.strip()]
    if not cleaned:
        return text

    vocab_lower = [v.lower() for v in cleaned]
    vocab_by_lower = dict(zip(vocab_lower, cleaned))

    def _match_case(source: str, replacement: str) -> str:
        # Preserve interior uppercase from the vocab entry (e.g. iPhone, MacDonald).
        has_interior_upper = any(ch.isupper() for ch in replacement[1:])
        if has_interior_upper:
            # Keep vocab casing, but uppercase the whole thing if source shouted.
            return replacement.upper() if source.isupper() else replacement
        if source.isupper():
            return replacement.upper()
        if source[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement.lower()

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        token_lower = token.lower()
        candidates = difflib.get_close_matches(
            token_lower, vocab_lower, n=1, cutoff=cutoff
        )
        if not candidates:
            return token
        # Guard against short-word false positives: only allow substitution
        # when the candidate is the same length or the token is long enough
        # that a one-character length difference is proportionally small.
        # For tokens of 4 chars or fewer, lengths must match exactly.
        candidate = candidates[0]
        max_len_diff = max(0, len(token_lower) // 5)
        if abs(len(token_lower) - len(candidate)) > max_len_diff:
            return token
        canonical = vocab_by_lower[candidate]
        return _match_case(token, canonical)

    return _WORD_RE.sub(_sub, text)
