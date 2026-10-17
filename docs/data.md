# Data and licences

Everything the notebooks read is the Hugging Face dataset
[`huseyin-karaca/fastt`](https://huggingface.co/datasets/huseyin-karaca/fastt).

## What it holds

| Folder | What |
|---|---|
| `<corpus>_base/` | the corpus's audio at 16 kHz mono with its reference transcripts: the clips the paper uses |
| `<corpus>_model_<expert>/` | one expert on one corpus, per clip: the transcript (raw and normalised), the word-error counters against the reference (substitutions, deletions, insertions, reference length), and the mean of the expert's final-layer encoder states |
| `results/<corpus>/` | the records: the main record (`main_<corpus>.json`), the stored 5×2 folds (`runlog/`), the ablation's exploration trials and its record, the timing clocks and record |
| `studies/` | the hyperparameter searches, one Optuna database per study |

LibriSpeech's folders carry no prefix (`base/`, `model_<expert>/`), and its records sit directly under `results/`. The audio of AMI, Earnings-22 and
People's Speech is also read by the companion project
[HIT-ASR](https://huseyin-karaca.github.io/hit-asr/), whose own dataset holds its experts' outputs.

## The corpora

| Corpus | Source | Used splits | Licence of everything derived from it |
|---|---|---|---|
| LibriSpeech | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) | dev-clean, dev-other, test-clean, test-other | CC BY 4.0 |
| VoxPopuli | [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) (`en`) | train (1 of 30 shards), validation, test | CC0 1.0 |
| AMI, single distant microphone | [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) (`sdm`) | train (4 of 27 shards), validation, test | CC BY 4.0 |
| Earnings-22 | [Rev](https://github.com/revdotcom/speech-datasets/tree/main/earnings22), via [distil-whisper/earnings22](https://huggingface.co/datasets/distil-whisper/earnings22) (`chunked`) | 8 of 38 shards: 4 for the hold-out and 4 for evaluation, one shard apart | CC BY-SA 4.0 |
| People's Speech | [MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech) (`dirty`, the CC BY part) | 1 train shard, 2 validation and 2 test shards | CC BY 2.0–4.0, per recording |

Each corpus's audio and the experts' outputs on it are distributed under that corpus's licence. The folders are kept
apart per corpus, so the dataset is a collection of separately licensed parts, not one combined work. In short:
**attribution** is required for all of them except VoxPopuli, and **ShareAlike** for Earnings-22 (derivatives under
the same licence). Please cite the corpora you use.

## The experts

The transcripts and features are outputs of these pretrained models; their licences allow redistributing them with
attribution.

| Expert | Model | Toolkit | Licence |
|---|---|---|---|
| Parakeet-TDT-0.6b-v2 | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | NeMo | CC BY 4.0 |
| Canary-Qwen-2.5b | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | NeMo | CC BY 4.0 |
| Whisper-large-v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 🤗 Transformers | Apache-2.0 |
| ESPnet Conformer-RNN-T | [espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp](https://huggingface.co/espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp) | ESPnet | CC BY 4.0 |
| ESPnet Transformer | [espnet/Shinji_Watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best](https://huggingface.co/espnet/Shinji_Watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best) | ESPnet | CC BY 4.0 |
| data2vec-audio-base | [facebook/data2vec-audio-base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h) | 🤗 Transformers | Apache-2.0 |
| wav2vec2-base-960h | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | 🤗 Transformers | Apache-2.0 |
| wav2vec2-large-960h | [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) | 🤗 Transformers | Apache-2.0 |

## The trios

| Corpus | The experts FASTT chooses among |
|---|---|
| LibriSpeech | Parakeet-TDT-0.6b-v2, Canary-Qwen-2.5b, Whisper-large-v3 |
| VoxPopuli | ESPnet Conformer-RNN-T, ESPnet Transformer, data2vec-audio-base |
| AMI (SDM) | ESPnet Conformer-RNN-T, ESPnet Transformer, wav2vec2-large-960h |
| Earnings-22 | data2vec-audio-base, ESPnet Conformer-RNN-T, wav2vec2-large-960h |
| People's Speech | data2vec-audio-base, ESPnet Transformer, wav2vec2-base-960h |

The configuration of every trio, and of everything else the paper reports, is fixed in `fastt.configs`.

## How the outputs were made

Every expert decodes every clip of its corpus once, from the 16 kHz audio. Its transcript is scored against the
reference after Whisper's English normaliser, giving the four word-error counters, and its final-layer encoder states
are averaged over time. `notebooks/extract` repeats exactly this pass from the published audio, and ends by comparing
the rebuilt transcripts and corpus WERs with the published ones.
