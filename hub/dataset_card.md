---
pretty_name: FASTT
license: other
license_name: per-corpus
license_link: https://huseyin-karaca.github.io/fastt/data/
language:
- en
task_categories:
- automatic-speech-recognition
tags:
- speech-recognition
- model-selection
- feature-learning
- encoder-representations
- reproducibility
---

# FASTT — data and results

The data behind **FASTT: Feature-Adaptive Speech-to-Text Model Selection** (Huseyin Karaca, A. Samil Namli,
Suleyman S. Kozat — Bilkent University, Turk Telekom 6G R&D Laboratory): what the pretrained speech recognizers of
the paper produce on its five English corpora, and the stored results every notebook of the code repository reads.

- **Code and notebooks:** [github.com/huseyin-karaca/fastt](https://github.com/huseyin-karaca/fastt)
- **Documentation:** [huseyin-karaca.github.io/fastt](https://huseyin-karaca.github.io/fastt)

## What is here

| Config / folder | What |
|---|---|
| `<corpus>_base` | the corpus's audio at 16 kHz mono and its reference transcripts: the clips the paper uses |
| `<corpus>_model_<expert>` | one expert on one corpus, per clip: its transcript (raw and Whisper-normalised), the word-error counters (substitutions, deletions, insertions, reference words) against the reference, and the mean of its final-layer encoder states |
| `results/<corpus>/` | the main record of each corpus (`main_<corpus>.json`: configuration, selected hyperparameters, every table), the stored 5×2 folds (`runlog/`), the ablation's records and the timing clocks |
| `studies/` | the hyperparameter searches, one Optuna database per study |

LibriSpeech's configurations carry no prefix (`base`, `model_<expert>`), and its records sit directly under
`results/`. A configuration loads with `datasets`, for example
`load_dataset("huseyin-karaca/fastt", "ami_sdm_model_wav2vec2_large")`; the package reads them through
`fastt.ensemble.EnsembleStore`. The audio of AMI, Earnings-22 and People's Speech is also read by the companion
project [HIT-ASR](https://huggingface.co/datasets/huseyin-karaca/hit-asr).

## The corpora and their licences

Every part keeps the licence of the corpus it is derived from; the configurations are kept apart per corpus, so this
dataset is a collection of separately licensed parts. **Attribution** is required for all of them except VoxPopuli,
and **ShareAlike** for Earnings-22.

| Corpus | Source | Licence |
|---|---|---|
| LibriSpeech (`base`, `model_*`) | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) | CC BY 4.0 |
| VoxPopuli, English (`voxpopuli_*`) | [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) | CC0 1.0 |
| AMI, single distant microphone (`ami_sdm_*`) | [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) | CC BY 4.0 |
| Earnings-22 (`earnings22_*`) | [Rev](https://github.com/revdotcom/speech-datasets/tree/main/earnings22), via [distil-whisper/earnings22](https://huggingface.co/datasets/distil-whisper/earnings22) | CC BY-SA 4.0 |
| People's Speech, `dirty` (CC BY) subset (`peoples_speech_*`) | [MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech) | CC BY 2.0–4.0, per recording |

The `results/` and `studies/` records are released under CC BY 4.0, except those of Earnings-22 (CC BY-SA 4.0).

## The experts

The transcripts and features are outputs of pretrained models, redistributed under their licences with attribution:

| Expert | Model | Licence |
|---|---|---|
| Parakeet-TDT-0.6b-v2 | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | CC BY 4.0 |
| Canary-Qwen-2.5b | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | CC BY 4.0 |
| Whisper-large-v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | Apache-2.0 |
| ESPnet Conformer-RNN-T | [espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp](https://huggingface.co/espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp) | CC BY 4.0 |
| ESPnet Transformer | [espnet/Shinji_Watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best](https://huggingface.co/espnet/Shinji_Watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best) | CC BY 4.0 |
| data2vec-audio-base | [facebook/data2vec-audio-base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h) | Apache-2.0 |
| wav2vec2-base-960h | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | Apache-2.0 |
| wav2vec2-large-960h | [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) | Apache-2.0 |

## The paper's trios

| Corpus | The experts FASTT chooses among |
|---|---|
| LibriSpeech | Parakeet-TDT-0.6b-v2, Canary-Qwen-2.5b, Whisper-large-v3 |
| VoxPopuli | ESPnet Conformer-RNN-T, ESPnet Transformer, data2vec-audio-base |
| AMI (SDM) | ESPnet Conformer-RNN-T, ESPnet Transformer, wav2vec2-large-960h |
| Earnings-22 | data2vec-audio-base, ESPnet Conformer-RNN-T, wav2vec2-large-960h |
| People's Speech | data2vec-audio-base, ESPnet Transformer, wav2vec2-base-960h |

## Citation

```bibtex
@article{karaca2026fastt,
  title  = {{FASTT}: Feature-Adaptive Speech-to-Text Model Selection},
  author = {Karaca, Huseyin and Namli, A. Samil and Kozat, Suleyman S.},
  year   = {2026},
  note   = {Under review at Expert Systems with Applications}
}
```

Please cite the corpora and the models you use as well.
