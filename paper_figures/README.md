# Intro figures

Run everything from the repository root, after `bash scripts/run.sh tables` (the scripts read the paper tables and
the per-image results of the tables' runs). Everything is written to `results/figures/`.

| Script | Command | GPU | Output |
|---|---|---|---|
| Fig. 1 (a): example images of each degradation shift | `python paper_figures/shift_crops.py` | no | `shift/<source>_to_<target>/*.png`, `shift/contact_*.png` |
| Fig. 1 (b): images of one test case | `GPU=0 bash scripts/run.sh figures` | yes | `qualitative/<set>/<image>/*.png`, `qualitative/contact_<set>.png` |
| Experiments: mean PSNR per backbone, pretrained -> ours | `python paper_figures/gain_per_backbone.py` | no | `gain_per_backbone.{pdf,png}` |
| Analysis: NSS space of source / target / simulated crops | `python paper_figures/nss_map.py --source_set SOTS_Outdoor --target OHAZY` | optional | `nss_map/*.{pdf,md}` |

## shift_crops.py

Exports evenly spaced degraded images of a source (training) test set and of its shifted test set, downsized to
`--max_side` px, plus a contact sheet per pair (source row above target row) to pick from. Default pairs: Rain100L ->
Rain100H, SOTS-Outdoor -> O-HAZY / DenseHaze, BSD68 σ25 -> Kodak24-Poisson, GoPro -> HIDE / DPDD, LOL -> LOLv2-S.
`--pairs A:B C:D` chooses pairs, `--n` images per set, `--crop S` exports random S x S crops instead of whole
images, `--gt` also saves the clean references.

## gain_per_backbone.py

One dumbbell per backbone and setting: mean PSNR over all test sets of the pretrained model (grey) and after ours
(blue). `--with_tta` adds existing TTA as a hollow orange marker. Reads the paper tables.

## nss_map.py

Projects the method's 32 NSS features of four groups into 2-D: crops of the source's own test set (the training
degradation), crops of the shifted target set, and clean DIV2K crops degraded by the simulator (8 candidates per clean
crop; the one closest to the target statistics is kept, exactly as the method selects it). The first call computes
the features (a few minutes) and caches them in a `.npz`; later calls with the same pair only redraw the plot.

| Option | Meaning | Default |
|---|---|---|
| `--source_set` | the source's own test set = the degradation the model was trained on | required |
| `--target` | the shifted test set | required |
| `--setting 3task` / `5task` | which families the simulator may use (3task: noise, rain, haze; 5task: + blur, low light) | 3task |
| `--proj pca` / `shift` | `pca`: plain 2-D principal components of source + target; `shift`: x = the direction separating source from target | pca |
| `--show_clean` | also draw the undegraded clean crops | off |
| `--recompute` | redo the features instead of using the cached `.npz` | off |

Pairs to try (source's own set -> shifted set):

```bash
python paper_figures/nss_map.py --source_set SOTS_Outdoor --target OHAZY                 # haze: synthetic -> real
python paper_figures/nss_map.py --source_set SOTS_Outdoor --target DenseHaze             # haze: synthetic -> dense real
python paper_figures/nss_map.py --source_set BSD68_s25 --target Kodak24_Poisson          # noise: Gaussian -> Poisson
python paper_figures/nss_map.py --source_set GoPro --target DPDD --setting 5task         # blur: motion -> defocus
python paper_figures/nss_map.py --source_set Rain100L --target Rain100H                  # rain: light -> heavy
```

The `.md` next to each plot gives the distance of every group to the target statistics (about 1 = a typical target
crop) and the families the selection kept. Use it to judge the plot: the story holds only if the selected simulated
crops are much closer to the target than the source crops are.

## qualitative.py (via `run.sh figures`)

`GPU=0 bash scripts/run.sh figures` does three things for one test case:
1. `select`: ranks the images of the set by how clearly ours beats both the pretrained model and existing TTA, from
   the per-image PSNRs the tables' runs already saved, and keeps 8 candidates plus the median-gain image.
2. Re-runs pretrained, existing TTA and ours on that set and saves only those images (`--save_names`; same runs and
   numbers as the tables, written to `results/figure_runs`).
3. `export`: per candidate, the full images (`input`, `pretrained`, `existing_tta`, `ours`, `gt`), the same with a
   zoom box (`*_box.png`), the zoom region (`*_zoom.png`) and `psnr.json`; plus a contact sheet of all candidates.

Choose the case with `FIG_BB` (backbone), `FIG_S` (3task / 5task) and `FIG_SET` (test set), e.g.
`FIG_BB=adair FIG_S=5task FIG_SET=DPDD GPU=0 bash scripts/run.sh figures`. After choosing an image from the contact
sheet, re-export it alone, optionally with a hand-set zoom square (x, y, size in pixels):
`python paper_figures/qualitative.py export --backbone adair --setting 5task --target SOTS_Indoor --image <name> --box 120,80,96`.

# Experiments figures

All of these follow `style.py` (ours blue circles, DCTTA orange triangles, pretrained grey; Times-like serif at
6.5-8 pt; sized to the column, 3.28 in, or the text width, 6.875 in, so include them without rescaling). Each script
writes a vector PDF and a PNG preview to `results/figures/`.

| Figure | Where | Command | GPU | Output |
|---|---|---|---|---|
| `fig:qualitative` | main, text width, 3.3 in tall | `GPU=0 bash scripts/run.sh qualitative` | yes | `qualitative_grid.pdf` |
| `fig:controls` | main, one column, 1.55 in tall | `python paper_figures/controls.py` | no | `controls.pdf` |
| `fig:cost` | main, one column, 1.6 in tall | `python paper_figures/cost.py --width column` | no | `cost_column.pdf` |
| (same, wide) | text width, if it moves back to the supplementary | `python paper_figures/cost.py` | no | `cost.pdf` |
| `fig:reliability` | supplementary, one column | `python paper_figures/reliability.py` | no | `reliability.pdf` |

`bash scripts/run.sh plots` runs the three table-driven scripts (after `run.sh tables`). Copies built from the Nov 9
tables are in `paper_figures/pdf/`; re-run `plots` whenever the tables change.

## qualitative_grid.py (via `run.sh qualitative`)

One test case per row: the full input with the zoom region marked, then the zoom of input, pretrained, + DCTTA,
+ ours and ground truth, with the full-image PSNR under each zoom.

```bash
GPU=0 bash scripts/run.sh qualitative       # the four rows of the paper (12 runs; DCTTA's take the longest)
```

For every row this (1) picks the five images whose gain over the pretrained model is closest to the median of the
set (`qualitative.py select --rule median`, from the tables' per-image PSNRs), (2) re-runs pretrained, ours and DCTTA
saving only those images to `results/figure_runs`, (3) composes the grid from the median-gain image of each row.
Finished runs are skipped, so the stage can be re-run.

Rows are `backbone:setting:set`, set with `QUAL_ROWS`; the output name with `QUAL_NAME`:

```bash
# default (main paper)
QUAL_ROWS="adair:5task:Rain100H moceir:3task:OHAZY bioir:3task:Kodak24_Poisson mirage:5task:DPDD"
# supplementary: the remaining target sets, including the two where ours does not gain
QUAL_ROWS="r2r:3task:SOTS_Indoor mirage:3task:DenseHaze promptir:5task:HIDE adair:5task:LOLv2_Syn" \
  QUAL_NAME=qualitative_supp GPU=0 bash scripts/run.sh qualitative
```

To change an image or a zoom region without re-running anything, call the composer directly. A row can name one of
the five saved images and a zoom region `x,y,w` (pixels of the evaluated image; the composer prints the ones it used):

```bash
python paper_figures/qualitative_grid.py \
  --rows adair:5task:Rain100H moceir:3task:OHAZY bioir:3task:Kodak24_Poisson mirage:5task:DPDD:<image>:300,180,160
python paper_figures/qualitative_grid.py --demo     # layout preview with synthetic images, no results needed
```

If a row was already run by `run.sh figures` (Fig. 1) with other saved names, delete that run's folder in
`results/figure_runs` first; the composer says which.

## controls.py, cost.py, reliability.py

- `controls.py`: one bar per control (mean change over the pretrained model, PromptIR three-task), grouped by what is
  adapted, plus the number of target sets each control improves. The 5000-step random initialisation stays in the
  caption.
- `cost.py`: three aligned panels (time, adapted parameters, mean change in PSNR), one row per backbone, DCTTA and
  ours joined by a line, every value printed.
- `reliability.py`: every backbone, setting and target set of the main tables as one mark, one row per target set.
  The axis has one visible break so that DCTTA's large losses are shown, not clipped (`--brk` moves it).
