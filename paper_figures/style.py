"""Shared figure style for the CVPR paper: page geometry, fonts, colours and table parsing.

CVPR geometry (cvpr.sty): \\textwidth 6.875 in, \\columnsep 0.3125 in -> one column 3.28 in. Fonts: Times in the
paper; matplotlib uses the first installed of Times New Roman / Times / Nimbus Roman / STIX (bundled with
matplotlib, Times-like). ``pdf.fonttype 42`` embeds TrueType instead of Type 3 fonts (PDF checkers flag Type 3).
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_W = 6.875
COL_W = (6.875 - 0.3125) / 2   # 3.28 in

# method identity (validated pair: CVD dE 24.7, normal-vision dE 33.6); ours = blue throughout the paper
C_OURS = '#2a78d6'
C_DCTTA = '#eb6834'
C_SOURCE = '#7a7a76'
INK, INK2, GRID = '#1f1f1d', '#5c5c58', '#d9d8d4'
# diverging (harm <-> gain): red / blue poles, neutral grey midpoint
DIV_NEG, DIV_MID, DIV_POS = '#c0392b', '#f0efec', '#1c5cab'

NAMES = {'promptir': 'PromptIR', 'adair': 'AdaIR', 'bioir': 'Bio-IR', 'moceir': 'MoCE-IR', 'mirage': 'MIRAGE',
         'r2r': 'R2R'}
TESTS = {'3task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson'],
         '5task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson', 'HIDE', 'LOLv2_Syn', 'DPDD']}
SHORT = {'Rain100H': 'Rain100H', 'SOTS_Indoor': 'SOTS-In', 'DenseHaze': 'DenseHaze', 'OHAZY': 'O-HAZY',
         'Kodak24_Poisson': 'Kodak24', 'HIDE': 'HIDE', 'LOLv2_Syn': 'LOLv2-S', 'DPDD': 'DPDD'}
FAMILY = {'Rain100H': 'rain', 'SOTS_Indoor': 'haze', 'DenseHaze': 'haze', 'OHAZY': 'haze', 'Kodak24_Poisson': 'noise',
          'HIDE': 'blur', 'LOLv2_Syn': 'low-light', 'DPDD': 'blur'}


def setup(fontsize=8):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Nimbus Roman No9 L', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 1, 'ytick.labelsize': fontsize - 1, 'legend.fontsize': fontsize - 1,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01, 'axes.unicode_minus': True,
    })


# ---------------------------------------------------------------------------- #
# tables
# ---------------------------------------------------------------------------- #
_INV_SHORT = {v: k for k, v in SHORT.items()}
_INV_NAMES = {v: k for k, v in NAMES.items()}
_NUM = re.compile(r'-?\d+(?:\.\d+)?')


def _num(cell):
    m = _NUM.search(cell.replace('−', '-'))
    return float(m.group()) if m else None


def parse_main_table(path):
    """``table_main_<s>.md`` (tools/table.py; also the hand-copied form with '+ DCTTA' rows) ->
    {(row, backbone, target): psnr}, row in source / dctta / ours. Cells may carry bold, ±std or /SSIM."""
    out, cols, bb = {}, None, None
    with open(path) as f:
        for line in f:
            if not line.startswith('|') or set(line.strip()) <= set('|-: '):
                continue
            cells = [c.strip().strip('*').strip() for c in line.strip().strip('|').split('|')]
            if cells[0] == 'Method':
                cols = [_INV_SHORT.get(c, c) for c in cells[1:]]
                continue
            if cols is None or cells[0] == 'Input':
                continue
            label = cells[0]
            if label.endswith('+ DCTTA'):
                row, name = 'dctta', label[:-len('+ DCTTA')].strip()
            elif label.endswith('+ Ours'):
                row, name = 'ours', label[:-len('+ Ours')].strip()
            else:
                row, name = 'source', label
            if name:
                bb = _INV_NAMES.get(name, name.lower())
            for c, v in zip(cols, cells[1:]):
                if c != 'Avg' and _num(v) is not None:
                    out[(row, bb, c)] = _num(v)
    return out


def load_main(root):
    """{(row, backbone, setting, target): psnr} from ``<root>/table_main_{3task,5task}.md``."""
    vals = {}
    for s in ('3task', '5task'):
        p = os.path.join(root, f'table_main_{s}.md')
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run `bash scripts/run.sh tables` first")
        for (r, bb, t), v in parse_main_table(p).items():
            vals[(r, bb, s, t)] = v
    return vals


def pairs(vals):
    """Backbone x setting x test-set keys present for all three rows, in table order."""
    out = []
    for s in ('3task', '5task'):
        for bb in NAMES:
            for t in TESTS[s]:
                if all((r, bb, s, t) in vals for r in ('source', 'dctta', 'ours')):
                    out.append((bb, s, t))
    return out
