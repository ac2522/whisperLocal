# Custom Vocabulary — Design

**Date:** 2026-04-20
**Status:** Approved
**Owner:** ac2522

## 1. Goal

Let users define domain-specific vocabulary (proper nouns, technical terms, names) that the app uses to improve transcription accuracy. Works on both Whisper and Parakeet, with a different mechanism per engine.

## 2. Mechanisms per engine

### Whisper — real prompt biasing
`pywhispercpp.Model.transcribe` accepts an `initial_prompt=` parameter (confirmed via `PARAMS_SCHEMA`). Whisper's decoder conditions on the prompt text, producing measurably better recognition of included words. The prompt has a hard token budget (~224 tokens in whisper.cpp); we cap ours conservatively at 200 characters' worth.

### Parakeet — post-transcript substitution
Parakeet (TDT) has no prompt-conditioning hook. We apply fuzzy whole-word substitution on the output: for each word in the transcript, find the best match in the vocab list above a similarity cutoff and replace. Case pattern of the original token is preserved. Uses `difflib` from stdlib — no new deps.

## 3. Limitations (documented in-app and in spec)

1. **Parakeet biasing is best-effort.** Words the model hears clearly but transcribes with a different phonetic approximation may not match the fuzzy rule. If Parakeet produces "Evrelo" for "Avrillo", the Levenshtein distance exceeds cutoff and no substitution happens.
2. **Whisper prompt length is capped** — long vocab lists are truncated at word boundaries with an INFO log. Hard ceiling ~200 characters.
3. **No phoneme-level biasing** on either engine. The best performance gain comes from Whisper's prompt mechanism; Parakeet improvements are incremental.
4. **Substitution can false-positive on short common words.** The cutoff (0.85) is conservative; users should avoid 2-4 character entries.
5. **Case preservation is heuristic.** "avrilo" → "avrillo", "Avrilo" → "Avrillo", "AVRILO" → "AVRILLO". Mixed-case in-word (e.g. "MacDonald") is preserved from the vocab entry.

## 4. Architecture

New module `engine/vocabulary.py` with two pure functions:

```python
def build_whisper_prompt(vocab: list[str], max_chars: int = 200) -> str | None: ...
def apply_post_substitution(text: str, vocab: list[str], cutoff: float = 0.85) -> str: ...
```

`build_whisper_prompt` joins entries with ", ", truncating at the last comma boundary that fits. Returns `None` for empty vocab (so callers can skip the kwarg).

`apply_post_substitution` walks word tokens via regex, runs `difflib.get_close_matches`, preserves case.

### Engine API change

Both `WhisperEngine.transcribe` and `ParakeetEngine.transcribe` gain a single new keyword-only parameter:

```python
def transcribe(self, audio_data, *, vocabulary: list[str] | None = None) -> str
```

Each engine routes `vocabulary` through its own mechanism internally. Call sites in `MainWindow` read the setting once per transcription and pass it.

### Settings

- New key: `custom_vocabulary: list[str]` (default `[]`).
- UI: one multi-line `QTextEdit` in the Settings dialog, new "Vocabulary" group box. One entry per line. Blank lines ignored. On save, split on newlines, strip each entry, drop empties.

### Wiring

`MainWindow._transcribe_and_emit` adds:
```python
vocabulary = self.settings.get("custom_vocabulary") or None
text = self.engine.transcribe(audio_data, vocabulary=vocabulary)
```

## 5. Errors

- Empty or whitespace-only vocab: `None` is passed to the engine, which skips its hook entirely. No error.
- Malformed entries (newlines inside a line): already split; no error.
- Whisper prompt overflow: silently truncate at word boundary, log INFO.

## 6. Tests (TDD, solid coverage)

New `tests/test_vocabulary.py`:

- `TestBuildWhisperPrompt` — empty list returns None; single entry returned as-is; multiple entries comma-joined; truncation respects word boundaries; returns None if first entry already exceeds budget.
- `TestApplyPostSubstitution` — exact match replaced with canonical vocab casing; fuzzy match above cutoff replaced; below cutoff left alone; short words (2-3 chars) don't false-positive against similar common words; case preservation (lower/title/upper) works; punctuation preserved; multiple substitutions in one string.

`tests/test_parakeet_engine.py` — add `TestTranscribeWithVocabulary`:
- vocabulary=None skips substitution
- vocabulary=["Avrillo"] substitutes "Avrilo" → "Avrillo"
- empty list treated as None

`tests/test_engine.py` (Whisper) — needs `@requires_model` gate already exists; add a mock-based test class to avoid requiring a real model:
- vocabulary=None → `transcribe` called without `initial_prompt` kwarg
- vocabulary=["Avrillo"] → `initial_prompt` kwarg contains "Avrillo"
- overlong vocab → kwarg is truncated, at word boundary

## 7. Out of scope

- Regex or pattern entries (only plain-word substitution)
- Phoneme-level biasing
- Per-mode vocabularies (one global list)
- Import/export of vocabulary files
