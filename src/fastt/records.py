"""The records the notebooks read: `results/<corpus>/<name>.json` (`results/<name>.json` for LibriSpeech) on the Hub.

Set `FASTT_RECORDS_DIR` to a folder holding records you wrote yourself (a run of `main_<corpus>` writes
`main_<corpus>.json` next to the notebook): a record found there is read instead of the Hub's.
"""

__all__ = ['load_record', 'records_dir']

import json
import os
from pathlib import Path

from fastt.hub import FasttHub


def records_dir():
    """The folder `FASTT_RECORDS_DIR` names, or `None`."""
    local = os.environ.get("FASTT_RECORDS_DIR")
    return Path(local) if local else None


def load_record(name, corpus=None):
    """The record `name` (`main_voxpopuli`, `timing_voxpopuli`, ...), parsed: from `FASTT_RECORDS_DIR` when it holds
    `<name>.json`, else `results/<corpus>/<name>.json` on the Hub (`results/<name>.json` without a corpus)."""
    from fastt.core import use_dataset
    local = records_dir()
    if local is not None and (local / f"{name}.json").exists():
        return json.loads((local / f"{name}.json").read_text())
    if corpus is not None:
        return FasttHub(spec=use_dataset(corpus, verbose=False)).load_results(name)
    return json.loads(Path(FasttHub().local_path(f"results/{name}.json")).read_text())
