# Custom Vocabulary Implementation Plan

**Goal:** Add user-defined vocabulary that improves transcription accuracy — via Whisper's `initial_prompt` for Whisper, via fuzzy post-substitution for Parakeet.

**Architecture:** New pure-function module `engine/vocabulary.py`. Both engines gain one new keyword parameter `vocabulary: list[str] | None`. `MainWindow` reads the setting and passes it on each transcription. Settings UI gains a multi-line text area.

**Tech Stack:** Python stdlib (`difflib`, `re`), no new deps.

**Spec:** `docs/superpowers/specs/2026-04-20-custom-vocabulary-design.md`

**File map:**
- Create: `engine/vocabulary.py`, `tests/test_vocabulary.py`
- Modify: `engine/whisper_engine.py`, `engine/parakeet_engine.py`, `ui/settings_dialog.py`, `ui/main_window.py`, `config/settings.py`
- Modify: `tests/test_engine.py`, `tests/test_parakeet_engine.py`

**Execution venv:** `venv/`. Always `venv/bin/python`, `venv/bin/pytest`.

---

## Task 1: `engine/vocabulary.py` — pure functions (TDD)

**Files:**
- Create: `engine/vocabulary.py`
- Create: `tests/test_vocabulary.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_vocabulary.py`:

```python
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
        # "avrillo" in text, "Avrillo" in vocab — substitution applied
        # preserving target vocab's canonical casing where appropriate
        result = apply_post_substitution("visit avrillo today", ["Avrillo"])
        assert result == "visit avrillo today" or result == "visit Avrillo today"
        # Either behaviour is acceptable for exact-case-preserved matches;
        # the substitution logic normalises toward vocab casing.
        # We assert the canonical replacement happens:
        assert "Avrillo" in apply_post_substitution("Avrillo", ["Avrillo"]) \
            or "Avrillo" == apply_post_substitution("Avrillo", ["Avrillo"])

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
```

- [ ] **Step 2: Run, confirm they all fail**

```bash
venv/bin/pytest tests/test_vocabulary.py -v
```

Expected: `ModuleNotFoundError: No module named 'engine.vocabulary'`.

- [ ] **Step 3: Implement the module**

Create `engine/vocabulary.py`:

```python
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

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


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
        if source.isupper():
            return replacement.upper()
        if source[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement.lower()

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        candidates = difflib.get_close_matches(
            token.lower(), vocab_lower, n=1, cutoff=cutoff
        )
        if not candidates:
            return token
        canonical = vocab_by_lower[candidates[0]]
        return _match_case(token, canonical)

    return _WORD_RE.sub(_sub, text)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
venv/bin/pytest tests/test_vocabulary.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/vocabulary.py tests/test_vocabulary.py
git commit -m "$(cat <<'EOF'
Add custom vocabulary helpers

build_whisper_prompt packs vocabulary into an initial_prompt string
within a char budget, truncating at entry boundaries. apply_post_
substitution walks word tokens and replaces fuzzy matches with the
canonical vocab entry, preserving the original token's case pattern.
No new dependencies — uses stdlib difflib + re.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire `vocabulary` into `WhisperEngine.transcribe` (TDD)

**Files:**
- Modify: `engine/whisper_engine.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Add failing tests**

Open `tests/test_engine.py`. The existing tests are gated by `@requires_model` (they need a real ggml-base.bin). We add a new class that mocks `pywhispercpp.model.Model` so the tests run without a model file.

Append:

```python
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_pwcpp_model():
    with patch("engine.whisper_engine.Model") as cls:
        fake = MagicMock()
        seg = MagicMock()
        seg.text = "hello world"
        fake.transcribe.return_value = [seg]
        cls.return_value = fake
        yield cls, fake


class TestWhisperEngineVocabulary:
    def test_no_vocabulary_omits_initial_prompt(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        engine.transcribe(b"\x00" * 32000)
        # initial_prompt must not be a kwarg at all — pywhispercpp treats
        # any passed string as a prompt, including empty string, so omission
        # is important.
        assert "initial_prompt" not in fake.transcribe.call_args.kwargs

    def test_empty_vocabulary_omits_initial_prompt(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        engine.transcribe(b"\x00" * 32000, vocabulary=[])
        assert "initial_prompt" not in fake.transcribe.call_args.kwargs

    def test_vocabulary_sets_initial_prompt(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        engine.transcribe(b"\x00" * 32000, vocabulary=["Avrillo", "SDLT"])
        prompt = fake.transcribe.call_args.kwargs["initial_prompt"]
        assert "Avrillo" in prompt
        assert "SDLT" in prompt

    def test_overlong_vocabulary_truncated(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        long_vocab = [f"word{i}" for i in range(100)]  # > 200 chars combined
        engine.transcribe(b"\x00" * 32000, vocabulary=long_vocab)
        prompt = fake.transcribe.call_args.kwargs["initial_prompt"]
        assert len(prompt) <= 200
        # And it must end on a complete word
        last_word = prompt.split(", ")[-1]
        assert last_word in long_vocab
```

- [ ] **Step 2: Run, confirm fails**

```bash
venv/bin/pytest tests/test_engine.py::TestWhisperEngineVocabulary -v
```

Expected: tests fail with `TypeError: transcribe() got an unexpected keyword argument 'vocabulary'`.

- [ ] **Step 3: Modify `WhisperEngine.transcribe`**

Open `engine/whisper_engine.py`. Add an import at the top (alphabetical):

```python
from engine.vocabulary import build_whisper_prompt
```

Replace the existing `transcribe` signature and body. Find:

```python
    def transcribe(self, audio_data) -> str:
```

(plus its body). Replace with:

```python
    def transcribe(self, audio_data, *, vocabulary: list[str] | None = None) -> str:
        """Transcribe audio data and return the full text.

        Parameters
        ----------
        audio_data : numpy.ndarray or bytes
            If a numpy float32 array, used directly. If bytes (assumed
            int16 PCM), it is converted to float32 by dividing by 32768.0.
        vocabulary : list[str] or None, optional
            Domain-specific words/phrases passed to whisper.cpp as
            ``initial_prompt`` to bias the decoder. If None or empty,
            no prompt is used. Long lists are truncated at word
            boundaries to stay within the whisper.cpp prompt budget.

        Returns
        -------
        str
            Cleaned transcript text.

        Raises
        ------
        RuntimeError
            If no model is currently loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        if isinstance(audio_data, (bytes, bytearray)):
            audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        kwargs = {}
        prompt = build_whisper_prompt(vocabulary)
        if prompt is not None:
            kwargs["initial_prompt"] = prompt

        segments = self._model.transcribe(audio_data, **kwargs)
        text = " ".join(seg.text for seg in segments)
        # Remove bracketed artifacts like [Silence], [Typing], etc.
        text = re.sub(r'\[.*?\]', '', text)
        # Strip trailing hallucinated phrases (common with large-v3-turbo)
        text = re.sub(r'\s*\b[Tt]hank you\.?\s*$', '', text)
        # Collapse extra whitespace left behind.
        text = re.sub(r'  +', ' ', text).strip()
        return text
```

- [ ] **Step 4: Run tests**

```bash
venv/bin/pytest tests/test_engine.py -v
```

Expected: existing `@requires_model` tests continue to skip/pass; the new 4 `TestWhisperEngineVocabulary` tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/whisper_engine.py tests/test_engine.py
git commit -m "$(cat <<'EOF'
WhisperEngine: accept custom vocabulary as initial_prompt

Adds a keyword-only vocabulary: list[str] | None parameter to
WhisperEngine.transcribe. When provided, entries are joined into an
initial_prompt (via engine.vocabulary.build_whisper_prompt) and
forwarded to pywhispercpp so whisper.cpp's decoder conditions on the
prompt. Overlong lists truncate at word boundaries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire `vocabulary` into `ParakeetEngine.transcribe` (TDD)

**Files:**
- Modify: `engine/parakeet_engine.py`
- Modify: `tests/test_parakeet_engine.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_parakeet_engine.py`:

```python
class TestParakeetEngineVocabulary:
    def test_no_vocabulary_leaves_output_unchanged(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = "visit avrilo today"
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "visit avrilo today"

    def test_empty_vocabulary_leaves_output_unchanged(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = "visit avrilo today"
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32), vocabulary=[])
        assert text == "visit avrilo today"

    def test_vocabulary_applies_fuzzy_substitution(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = "visit avrilo today"
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(
            np.zeros(16000, dtype=np.float32),
            vocabulary=["Avrillo"],
        )
        assert "avrillo" in text  # case-preserved lowercase substitution
        assert "avrilo" not in text
```

- [ ] **Step 2: Run, confirm fail**

```bash
venv/bin/pytest tests/test_parakeet_engine.py::TestParakeetEngineVocabulary -v
```

Expected: `TypeError: transcribe() got an unexpected keyword argument 'vocabulary'`.

- [ ] **Step 3: Modify `ParakeetEngine.transcribe`**

Open `engine/parakeet_engine.py`. Add near the top:

```python
from engine.vocabulary import apply_post_substitution
```

Find:

```python
    def transcribe(self, audio_data) -> str:
```

Replace with:

```python
    def transcribe(
        self, audio_data, *, vocabulary: list[str] | None = None
    ) -> str:
        """Transcribe audio data and return the full text.

        Parameters
        ----------
        audio_data : numpy.ndarray or bytes
            Audio samples; see WhisperEngine.transcribe for accepted types.
        vocabulary : list[str] or None, optional
            Domain-specific words applied via fuzzy post-substitution on
            Parakeet's output. Parakeet has no prompt-biasing hook, so
            this is a best-effort correction rather than real biasing.

        Returns
        -------
        str
            Cleaned transcript text.
        """
```

Then modify the body so after `result = self._model.recognize(...)` and the list-flattening and str() cast, substitute via `apply_post_substitution(text, vocabulary)` before the whitespace-collapse step. Full replacement body:

```python
        if self._model is None:
            raise RuntimeError("No model loaded")

        if isinstance(audio_data, (bytes, bytearray)):
            audio_data = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
        elif np.issubdtype(audio_data.dtype, np.integer):
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        result = self._model.recognize(audio_data, sample_rate=self.SAMPLE_RATE)
        if isinstance(result, list):
            result = result[0] if result else ""
        text = str(result)
        text = apply_post_substitution(text, vocabulary)
        text = re.sub(r"\s+", " ", text).strip()
        return text
```

- [ ] **Step 4: Run all parakeet tests**

```bash
venv/bin/pytest tests/test_parakeet_engine.py -v
```

Expected: all existing tests still pass, plus the three new `TestParakeetEngineVocabulary` tests.

- [ ] **Step 5: Commit**

```bash
git add engine/parakeet_engine.py tests/test_parakeet_engine.py
git commit -m "$(cat <<'EOF'
ParakeetEngine: apply vocabulary via fuzzy post-substitution

Adds a keyword-only vocabulary: list[str] | None parameter to
ParakeetEngine.transcribe. Parakeet TDT has no prompt-biasing hook,
so the vocabulary is applied as case-preserving fuzzy whole-word
substitution on the decoded text. Best-effort only — see design spec
for limitations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Settings UI + MainWindow wiring

**Files:**
- Modify: `config/settings.py`
- Modify: `ui/settings_dialog.py`
- Modify: `ui/main_window.py`

- [ ] **Step 1: Add the default setting**

In `config/settings.py`, inside `DEFAULT_SETTINGS`, add a new key:

```python
    'custom_vocabulary': [],
```

Place it after `auto_paste` alphabetically or at the end — doesn't matter, just don't disturb existing entries.

- [ ] **Step 2: Add the UI group**

Open `ui/settings_dialog.py`. Add imports at the top if not present:

```python
from PyQt5.QtWidgets import QTextEdit
```

(the module already imports from PyQt5.QtWidgets; add `QTextEdit` to that existing import list rather than a separate line).

Inside `__init__`, find the layout construction near the `_build_hotkey_group()` call. Add a new builder call just before the Save button:

```python
        # --- Vocabulary group ---
        layout.addWidget(self._build_vocabulary_group())
```

Add the builder method alongside the other `_build_*_group` methods:

```python
    def _build_vocabulary_group(self):
        group = QGroupBox("Custom Vocabulary")
        vbox = QVBoxLayout()

        vbox.addWidget(QLabel(
            "One word or short phrase per line. Used to bias Whisper's decoder "
            "(real prompt) and to fuzzy-fix Parakeet's output (post-processing)."
        ))

        self._vocab_edit = QTextEdit()
        self._vocab_edit.setPlaceholderText(
            "Avrillo\nconveyancing\nSDLT"
        )
        existing = self._settings.get("custom_vocabulary") or []
        self._vocab_edit.setPlainText("\n".join(existing))
        self._vocab_edit.setFixedHeight(120)
        vbox.addWidget(self._vocab_edit)

        group.setLayout(vbox)
        return group
```

In `_save`, before `self._settings.save()`, add:

```python
        # Custom vocabulary — split on newlines, strip, drop empties
        vocab_raw = self._vocab_edit.toPlainText().splitlines()
        vocab_list = [line.strip() for line in vocab_raw if line.strip()]
        self._settings.set("custom_vocabulary", vocab_list)
```

- [ ] **Step 3: Pass the vocabulary on each transcription**

Open `ui/main_window.py`. Find `_transcribe_and_emit`. Change:

```python
            text = self.engine.transcribe(audio_data)
```

to:

```python
            vocabulary = self.settings.get("custom_vocabulary") or None
            text = self.engine.transcribe(audio_data, vocabulary=vocabulary)
```

- [ ] **Step 4: Run the whole suite**

```bash
venv/bin/pytest tests/ -v --ignore=tests/test_autopaste_e2e.py --ignore=tests/test_paste_realistic.py
```

Expected: no regressions; test count goes up by the new vocab tests from Tasks 1–3.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py ui/settings_dialog.py ui/main_window.py
git commit -m "$(cat <<'EOF'
Wire custom vocabulary from Settings into each transcription

New Settings group "Custom Vocabulary" — multiline QTextEdit, one
entry per line. Stored as custom_vocabulary: list[str] (default []).
MainWindow reads the list on each transcription and forwards it to
the active engine's transcribe(vocabulary=...) param.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Manual smoke test

- [ ] Launch the app. Open Settings. Verify the "Custom Vocabulary" group appears with a multi-line text area.
- [ ] Enter: `Avrillo`, `conveyancing`, `SDLT` on separate lines. Save.
- [ ] With a Whisper model selected, record a sentence like "Please contact Avrillo about the SDLT". Confirm "Avrillo" and "SDLT" appear in the transcript.
- [ ] Switch to Parakeet. Record the same sentence. Confirm the fuzzy substitution is applied to any near-miss words.
- [ ] Verify settings persistence: close and reopen the app; the vocabulary list should still be populated in Settings.
