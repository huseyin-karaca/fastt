__all__ = ['load_english_normalizer', 'WerScorer', 'corpus_wer_table']

import jiwer
import pandas as pd


def load_english_normalizer(model_id="openai/whisper-large-v3"):
    """Return the Whisper English text normaliser used by the model cards.

    Beyond case and punctuation it folds spelled-out numbers to digits,
    British/American spellings and contractions — applied to BOTH sides, so
    "TWENTY" and "20" agree.

    Degrades gracefully: without `transformers` you get an uppercasing
    normaliser instead, which is enough for the CTC models whose references are
    already uppercase, but will read high for Whisper/Parakeet/Canary.
    """
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError:
        print("WARNING: transformers not installed; falling back to a naive "
              "uppercase normaliser. `wer` will not match the model cards.")
        return lambda s: " ".join(s.upper().split())

    mapping = {}
    try:
        from transformers import WhisperTokenizer
        mapping = WhisperTokenizer.from_pretrained(model_id).english_spelling_normalizer
    except Exception as e:                                 # noqa: BLE001
        print(f"WARNING: could not load the spelling map ({type(e).__name__}); "
              "normalising without spelling harmonisation. WER will read "
              "slightly high versus the model cards.")
    return EnglishTextNormalizer(mapping)


class WerScorer:
    """Word-level error counters, computed twice per utterance.

    Parameters
    ----------
    normalizer : any `str -> str` callable. Defaults to the Whisper English
        normaliser; pass your own to avoid the `transformers` dependency.
    """

    def __init__(self, normalizer=None):
        self.normalizer = normalizer if normalizer is not None else load_english_normalizer()

    @staticmethod
    def _counters(ref, hyp):
        """(substitutions, deletions, insertions, reference word count)."""
        if not ref.strip():
            return 0, 0, len(hyp.split()), 0
        w = jiwer.process_words(ref, hyp)
        # N = hits + substitutions + deletions = total words in the reference
        return w.substitutions, w.deletions, w.insertions, (
            w.hits + w.substitutions + w.deletions
        )

    def score(self, refs, hyps):
        """Score a batch. Returns a dict of columns ready to merge into a batch."""
        refs_n = [self.normalizer(r) for r in refs]
        hyps_n = [self.normalizer(h) for h in hyps]

        cols = {k: [] for k in (
            "sub", "dele", "ins", "nref", "wer",
            "sub_raw", "dele_raw", "ins_raw", "nref_raw", "wer_raw",
        )}
        for ref, hyp, ref_n, hyp_n in zip(refs, hyps, refs_n, hyps_n):
            for suffix, (r, h) in (("", (ref_n, hyp_n)), ("_raw", (ref, hyp))):
                s, d, i, n = self._counters(r, h)
                cols[f"sub{suffix}"].append(s)
                cols[f"dele{suffix}"].append(d)
                cols[f"ins{suffix}"].append(i)
                cols[f"nref{suffix}"].append(n)
                # Utterance-level WER. For a corpus figure use sum(S+D+I)/sum(N),
                # never the mean of this column.
                cols[f"wer{suffix}"].append((s + d + i) / n if n > 0 else 0.0)

        cols["text_norm"] = refs_n
        cols["transcription_norm"] = hyps_n
        return cols


def corpus_wer_table(ds):
    """Corpus WER per split, for both the normalised and the raw scoring.

    Returns a DataFrame with one row per split. This is the table to check
    against a model card before pushing anything.
    """
    rows = []
    for split, d in ds.items():
        row = {"split": split, "n_utts": d.num_rows}
        for suffix, label in (("", "norm"), ("_raw", "raw")):
            err = sum(d[f"sub{suffix}"]) + sum(d[f"dele{suffix}"]) + sum(d[f"ins{suffix}"])
            ref = sum(d[f"nref{suffix}"])
            row[f"wer_{label}"] = err / ref if ref else float("nan")
            row[f"errors_{label}"] = err
            row[f"ref_words_{label}"] = ref
        rows.append(row)
    return pd.DataFrame(rows)
