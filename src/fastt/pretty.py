__all__ = ['Table', 'show', 'HEX', 'FAMILIES', 'STATES', 'in_notebook', 'use_colors', 'colors_enabled', 'paint', 'family', 'metric_token', 'signed', 'state', 'heading', 'table_html', 'table_text', 'show_table']

import html as _html
import os
import re
import sys

import numpy as np
import pandas as pd

# `dim` is faint AND grey: Colab's renderer ignores SGR 2 on its own.
_ANSI = {
    "bold": "1", "dim": "2;90", "italic": "3", "underline": "4",
    "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
    "cyan": "36", "grey": "90", "white": "97",
}
# The same names as CSS colours — mid-luminance, so they read on Colab's light
# theme and on its dark one.
HEX = {"red": "#d9534f", "green": "#2f9e5f", "yellow": "#c98a12", "blue": "#3b7dd8",
       "magenta": "#a35bc4", "cyan": "#1f93b0", "grey": "#8a8f98"}

# Metric families, matched on either the manuscript header (`GC@1`) or the
# registry name (`gap_closed_tol1`), so one palette serves every table.
FAMILIES = (
    ("wer",  re.compile(r"^(WER|corpus_wer|wer)$"),                       ("bold",),        None),
    ("uwer", re.compile(r"^(uWER|mean_utt_wer)$"),                        ("bold", "blue"), "blue"),
    ("gc",   re.compile(r"^(GC@\d|gap_closed(_tol\d)?)$"),             ("green",),       "green"),
    ("ugc",  re.compile(r"^(uGC@\d|gap_closed_mean_utt(_tol\d)?)$"),       ("cyan",),        "cyan"),
    ("acc",  re.compile(r"^(Acc@\d|acc(_tol\d|_strict|_lenient)?)$"),      ("magenta",),     "magenta"),
    ("dist", re.compile(r"^(dist|selection_dist)$"),                       ("grey",),        "grey"),
)
# Lower is better for the WERs, higher for everything else that has a family.
_MINIMISE = {"wer", "uwer"}

# Words that carry a state, and the colour each one gets wherever it appears.
STATES = {"complete": "green", "partial": "yellow", "absent": "red",
          "HIT": "green", "MISS": "yellow", "WROTE": "blue", "FAILED": "red",
          "pruned": "yellow", "reused": "green", "adopted": "cyan"}

_COLOR = None            # None = decide from the environment; True / False force it


def in_notebook():
    """True under a Jupyter or Colab kernel — where HTML can be displayed."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:                                      # noqa: BLE001
        return False
    return ip is not None and ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell")


def use_colors(flag=None):
    """Force ANSI colour on (`True`), off (`False`) or back to automatic (`None`)."""
    global _COLOR
    _COLOR = flag


def colors_enabled():
    """Whether `paint` emits escape codes: forced, else notebook or a TTY, never under `NO_COLOR`."""
    if _COLOR is not None:
        return bool(_COLOR)
    if os.environ.get("NO_COLOR"):
        return False
    if in_notebook():
        return True
    out = sys.stdout
    return bool(getattr(out, "isatty", None) and out.isatty())


def paint(text, *styles):
    """`text` wrapped in ANSI codes for `styles` (names in `_ANSI`); the bare text when colour is off."""
    text = str(text)
    if not styles or not colors_enabled():
        return text
    return "\x1b[" + ";".join(_ANSI[s] for s in styles) + "m" + text + "\x1b[0m"


def family(name):
    """`(family, ansi styles, css colour name)` for a metric header or registry name; `None` otherwise."""
    for fam, rx, ansi, css in FAMILIES:
        if rx.match(str(name)):
            return fam, ansi, css
    return None


def metric_token(label, value):
    """`label value` with the label dimmed and the value in its family's colour."""
    fam = family(label)
    return paint(label, "dim") + " " + (paint(value, *fam[1]) if fam else str(value))


def signed(text, value):
    """`text` green when `value` is positive, red when negative, plain otherwise."""
    if value is None or not np.isfinite(value) or value == 0:
        return str(text)
    return paint(text, "green" if value > 0 else "red")


def state(word):
    """A state word (`complete`, `HIT`, `MISS`, ...) in its colour, bold."""
    colour = STATES.get(str(word).strip())
    return paint(word, colour, "bold") if colour else str(word)


def heading(text, level=1):
    """A bold heading (level 1) or a dim sub-heading (level 2), as one string."""
    return paint(text, "bold") if level == 1 else paint(text, "dim")


# ------------------------------------------------------------------ tables --

_NUM = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)(?:\s*[±]\s*\d+(?:\.\d+)?)?\s*%?\*?\s*$")
_LEAD = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)")
_MISSING = ("", "—", "---", "nan", "NaN", "None")

_CSS = """
.ftw{font:12.5px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:inherit;margin:6px 0 14px}
.ftw .title{font-weight:600;font-size:14px;padding:2px 0 2px}
.ftw .sub{font-size:12px;opacity:.7;max-width:960px;padding-bottom:6px;line-height:1.4}
.ft{border-collapse:collapse;font:inherit;color:inherit;margin:2px 0 6px}
.ft th{font-weight:600;text-align:right;padding:5px 9px;border-bottom:2px solid rgba(128,128,128,.5);white-space:nowrap;background:rgba(128,128,128,.10);vertical-align:bottom}
.ft th.l,.ft td.l{text-align:left}
.ft td{padding:3px 9px;border-bottom:1px solid rgba(128,128,128,.18);white-space:nowrap;font-variant-numeric:tabular-nums;vertical-align:top}
.ft td.w{white-space:normal;max-width:720px}
.ft tr.g td{background:rgba(128,128,128,.16);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;opacity:.9;padding:4px 9px;border-top:1px solid rgba(128,128,128,.5)}
.ft tr.hl td{background:rgba(47,158,95,.11)}
.ft td.b{font-weight:700}
.ft .m{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px}
.ft th.f,.ft td.f{border-left:1px solid rgba(128,128,128,.4)}
.ft td.na{opacity:.45;text-align:center}
.ftw .note{font-size:11.5px;opacity:.75;max-width:960px;line-height:1.4;margin:2px 0 4px}
"""


def _fmt(v, decimals=None):
    """One cell as text: `—` for missing, floats to `decimals` places, `str` otherwise.

    With `decimals=None` a float keeps up to four places, trailing zeros trimmed.
    """
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if decimals is not None:
            return f"{float(v):.{decimals}f}"
        return str(int(v)) if float(v).is_integer() else f"{float(v):.4f}".rstrip("0").rstrip(".")
    return str(v)


def _decimals(values):
    """Places every float in a column is shown with: the most any of them needs, at most four.

    One rule per column, so `100.0` and `137.9` in the same column read as `100.0`
    and `137.9` rather than as `100` and `137.9`.
    """
    need = 0
    for v in values:
        if isinstance(v, (float, np.floating)) and np.isfinite(v) and not isinstance(v, bool):
            s = repr(float(v))
            need = max(need, 4 if "e" in s else min(4, len(s.split(".")[1].rstrip("0"))))
    return need


def _column_formatter(values):
    """`cell -> text` for one column, floats at the column's shared number of places."""
    d = _decimals(values)
    return lambda v: _fmt(v, d) if isinstance(v, (float, np.floating)) else _fmt(v)


def _lead(v):
    """The leading number of a cell (`0.4011 ± 0.0029` -> 0.4011), or NaN."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    m = _LEAD.match(str(v))
    return float(m.group(1)) if m else np.nan


def _numeric_column(values):
    """Right-align a column when every non-missing cell reads as a number or `mean ± sd`."""
    seen, fmt = False, _column_formatter(values)
    for v in values:
        s = fmt(v)
        if s in _MISSING:
            continue
        seen = True
        if not _NUM.match(s):
            return False
    return seen


def _best_cells(df, best, exclude):
    """`{(row position, column): True}` for the best value of each column in `best`."""
    out = {}
    if not best:
        return out
    if best == "auto":
        best = {c: ("min" if family(c)[0] in _MINIMISE else "max")
                for c in df.columns if family(c) and family(c)[0] != "dist"}
    for col, how in best.items():
        if col not in df.columns:
            continue
        vals = []
        for i, (_, row) in enumerate(df.iterrows()):
            if exclude is not None and exclude(row):
                continue
            x = _lead(row[col])
            if np.isfinite(x):
                vals.append((i, x))
        if not vals:
            continue
        target = (min if how == "min" else max)(x for _, x in vals)
        for i, x in vals:
            if x == target:
                out[(i, col)] = True
    return out


def table_html(df, title=None, caption=None, note=None, group=None, best=None,
               best_exclude=None, highlight=None, mono=(), wrap=(), cell_style=None,
               header_style=None, index=False):
    """The HTML `show_table` displays. See there for the parameters."""
    df = df.reset_index() if index else df
    body_cols = [c for c in df.columns if c != group]
    numeric = {c: _numeric_column(df[c]) for c in body_cols}
    fmt = {c: _column_formatter(df[c]) for c in body_cols}
    bests = _best_cells(df, best, best_exclude)
    fams = [family(c) for c in body_cols]
    esc = _html.escape

    head = []
    for j, c in enumerate(body_cols):
        cls = [] if numeric[c] else ["l"]
        if fams[j] and (j == 0 or fams[j - 1] is None or fams[j - 1][0] != fams[j][0]):
            cls.append("f")
        css = []
        if fams[j] and fams[j][2]:
            css.append(f"color:{HEX[fams[j][2]]}")
        if header_style:
            extra = header_style(c)
            if extra:
                css.append(extra)
        head.append(f'<th class="{" ".join(cls)}" style="{";".join(css)}">{esc(str(c))}</th>')

    rows, last_group = [], object()
    for i, (_, row) in enumerate(df.iterrows()):
        if group is not None and row[group] != last_group:
            last_group = row[group]
            rows.append(f'<tr class="g"><td class="l" colspan="{len(body_cols)}">'
                        f'{esc(_fmt(last_group))}</td></tr>')
        tr_cls = "hl" if highlight is not None and highlight(row) else ""
        cells = []
        for j, c in enumerate(body_cols):
            v = row[c]
            s = fmt[c](v)
            cls = [] if numeric[c] else ["l"]
            if fams[j] and (j == 0 or fams[j - 1] is None or fams[j - 1][0] != fams[j][0]):
                cls.append("f")
            if s in _MISSING:
                cls.append("na")
            if c in mono:
                cls.append("m")
            if c in wrap:
                cls.append("w")
            if bests.get((i, c)):
                cls.append("b")
            css = []
            if fams[j] and fams[j][2] and s not in _MISSING:
                css.append(f"color:{HEX[fams[j][2]]}")
            if cell_style:
                extra = cell_style(c, v, row)
                if extra:
                    css.append(extra)
            cells.append(f'<td class="{" ".join(cls)}" style="{";".join(css)}">{esc(s)}</td>')
        rows.append(f'<tr class="{tr_cls}">' + "".join(cells) + "</tr>")

    def para(text, cls):
        return (f'<div class="{cls}">' + esc(str(text)).replace("\n", "<br>") + "</div>"
                if text else "")

    # Title and caption sit outside the `<table>` — a `<caption>` is as wide as
    # its table and wraps a sentence into a column beside a narrow one.
    return (f'<style>{_CSS}</style><div class="ftw">{para(title, "title")}{para(caption, "sub")}'
            f'<div style="overflow-x:auto"><table class="ft">'
            f"<thead><tr>{''.join(head)}</tr></thead><tbody>{''.join(rows)}</tbody>"
            f'</table></div>{para(note, "note")}</div>')


def table_text(df, title=None, caption=None, note=None, group=None, index=False, color=True):
    """The plain-text rendering `show_table` prints outside a notebook.

    `color=False` gives the bare text — what goes into the `text/plain` twin of an
    HTML table, where escape codes would only litter the saved notebook.
    """
    p = paint if color else (lambda t, *s: str(t))
    lines = []
    if title:
        lines.append(p(title, "bold"))
    if caption:
        lines.append(p(caption, "dim"))
    if group is not None and group in df.columns:
        for g, sub in df.groupby(group, sort=False):
            lines.append(p(f"  [{_fmt(g)}]", "bold"))
            lines.append(sub.drop(columns=[group]).to_string(index=index))
    else:
        lines.append(df.to_string(index=index))
    if note:
        lines.append(p(note, "dim"))
    return "\n".join(lines)


def show_table(df, title=None, caption=None, note=None, group=None, best=None,
               best_exclude=None, highlight=None, mono=(), wrap=(), cell_style=None,
               header_style=None, index=False):
    """Display a frame as a styled table in a notebook; print it as text anywhere else.

    Parameters
    ----------
    title, caption : bold line above the table and a dim one under it.
    note : small paragraph under the table (a caveat that must travel with it).
    group : a column whose runs of equal values become group-header rows; the
        column itself is not shown.
    best : `{column: "min" | "max"}` — the best value per column is bold; or
        `"auto"`, which reads the direction off the metric family (WERs down,
        gaps and accuracies up).
    best_exclude : `row -> bool`; rows for which it is true do not compete for
        "best" (the oracle rows of Table 3, which are 100% by construction).
    highlight : `row -> bool`; rows for which it is true are tinted.
    mono, wrap : columns set in monospace, and columns allowed to wrap.
    cell_style, header_style : `(column, value, row) -> css` and `column -> css`
        hooks for one table's own rules (p-value shading, say).
    index : show the index as the first column.

    Numeric columns — plain numbers, `mean ± sd`, percentages — are right-aligned;
    `—` cells are dimmed; metric columns take their family colour and each family
    opens with a thin rule. Returns `None` so the frame is not shown twice when
    the call is the last expression of a cell.
    """
    if in_notebook():
        from IPython.display import display

        display({"text/html": table_html(df, title, caption, note, group, best, best_exclude,
                                         highlight, mono, wrap, cell_style, header_style,
                                         index),
                 "text/plain": table_text(df, title, caption, note, group, index, color=False)},
                raw=True)
    else:
        print(table_text(df, title, caption, note, group, index), flush=True)


class Table:
    """A frame with its title, caption and styling: displayed as `show_table` displays it when it is the value a
    notebook cell ends on, printed as text anywhere else. `.df` is the frame itself."""

    def __init__(self, df, title=None, caption=None, note=None, **style):
        self.df, self.title, self.caption, self.note, self.style = df, title, caption, note, style

    def _repr_html_(self):
        return table_html(self.df, self.title, self.caption, self.note, **self.style)

    def __repr__(self):
        return table_text(self.df, self.title, self.caption, self.note, self.style.get("group"),
                          self.style.get("index", False), color=False)

    def show(self):
        show_table(self.df, self.title, self.caption, self.note, **self.style)


def show(*tables):
    """Display several `Table`s from one cell, in order."""
    for t in tables:
        t.show()
