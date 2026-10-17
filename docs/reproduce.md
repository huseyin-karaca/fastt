# Reproduce the paper

Every notebook runs at one of three levels; set `LEVEL` in its first code cell. The
[reproducibility report](reproducibility.md) says what we have checked each level against.

| Level | What runs | What it reads | Needs |
|---|---|---|---|
| **1** | every table | the searches, folds and records stored on the Hub, the experts' error counters | a CPU, minutes |
| **2** | every fold refitted; in the main notebooks `RETUNE = True` runs the hyperparameter search again too | the experts' features and error counters | a CPU, hours |
| **3** | level 2, on features you rebuilt from the audio | the local folder `extract` wrote, on the same machine | a GPU for `extract` |

Every level reads only the public dataset and needs no account and no token. Nothing is ever written to the Hub.

## Level 1: the tables from the stored results

Open a notebook in Colab (the badges on the [notebooks page](notebooks.md)) and run it top to bottom; any runtime will
do. A main notebook downloads the three experts' error counters of its corpus and the stored searches and folds, and
shows Tables 1, 2, 3 and 6.

Level 1 **trains nothing**. A search or fold missing from the Hub stops it with an error rather than training it
quietly.

## Level 2: refit every fold

Set `LEVEL = 2`. The experts' pooled features are downloaded and every method is trained again on every fold, on the
same folds. With `RETUNE = True` the hyperparameter search runs again too, ten random draws per method on the
hold-out. The notebook keeps what it computes on the local disk (`$FASTT_CACHE`, by default `~/.cache/fastt`) and
writes its record, `main_<corpus>.json`, next to itself.

To read your own records in the `ablation` and `manuscript_tables` notebooks, set `FASTT_RECORDS_DIR` to the folder
that holds them before the imports.

## Level 3: rebuild the features from the audio

`notebooks/extract` decodes every corpus with each expert of its trio, from the audio in the published dataset, and
stores their transcripts, word-error counters and pooled encoder states in a local folder, `$FASTT_CACHE/rebuilt`.

The experts come from three toolkits that do not install into one environment: **A** 🤗 Transformers (wav2vec2,
data2vec, Whisper), **B** NeMo (Parakeet-TDT, Canary-Qwen) and **C** ESPnet. LibriSpeech's trio needs groups A and B,
the other four corpora's trios groups A and C.

1. Open `extract` on a GPU runtime, set `GROUP` and `CORPORA`, and run it. Repeat in a fresh runtime for each group
   the trios need; an expert already complete in the folder is skipped. It ends by comparing your transcripts and
   WERs with the published ones.
2. On the same machine, run a main notebook at `LEVEL = 3`: it reads the features from that folder and the published
   searches from the Hub, and refits every fold.

To run level 3 elsewhere, copy the folder and set `FASTT_HUB=local:/path/to/it` before the main notebook's imports.

## Locally

```bash
pip install "fastt[paper] @ git+https://github.com/huseyin-karaca/fastt"
git clone https://github.com/huseyin-karaca/fastt && cd fastt
jupyter lab notebooks/main_ami_sdm.ipynb
```

Level 1 runs on a laptop. A main notebook also runs as a script, with the tables printed as text:

```bash
python -m fastt.experiment ami_sdm                  # level 1
python -m fastt.experiment ami_sdm --level 2        # every fold refitted
```

With PyTorch and XGBoost in one process, set `OMP_NUM_THREADS=1` if a run stops with a segmentation fault; on some
machines their two OpenMP runtimes do not coexist.

## Docker

An image on the GitHub container registry, built from `docker/` by the repository's `docker` workflow, holds the
package and the notebooks with a CPU build of PyTorch, keeps what it downloads in `/cache` and writes the records to
`/out`:

```bash
docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/fastt:cpu                                   # AMI, level 1
docker run --rm -v "$PWD/out:/out" ghcr.io/huseyin-karaca/fastt:cpu python -m fastt.experiment voxpopuli
docker run --rm -v "$PWD/out:/out" -v "$PWD/cache:/cache" ghcr.io/huseyin-karaca/fastt:cpu \
    python -m fastt.experiment voxpopuli --level 2
```

`docker build -f docker/Dockerfile -t fastt:cpu .` builds it locally.
