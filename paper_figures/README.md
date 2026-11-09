# Intro figures

Run everything from the repository root, after `bash scripts/run.sh tables` (the scripts read the paper tables and
the per-image results of the tables' runs). Everything is written to `results/figures/`.

| Figure | Command | GPU | Output |
|---|---|---|---|
| Fig. 2 (right), option A: summary bars | `python paper_figures/summary_bars.py` | no | `summary_bars.pdf`, `figure_stats.md` |
| Fig. 2 (right), option B: NSS map | `python paper_figures/nss_map.py --source_set SOTS_Outdoor --target OHAZY` | optional | `nss_map/SOTS_Outdoor_to_OHAZY.{pdf,md}`, `nss_map/examples/` |
| Fig. 1 (left): images of one test case | `GPU=0 bash scripts/run.sh figures` | yes | `qualitative/<set>/<image>/*.png`, `qualitative/contact_<set>.png` |

## summary_bars.py

One group per degradation family; the zero line is the pretrained model, bars are the mean PSNR change of existing
AiO TTA and of ours over the six backbones and both settings, dots are single backbone x setting results.
`figure_stats.md` holds the numbers for the abstract and intro (cells harmed, average gains per backbone).

| Option | Meaning | Default |
|---|---|---|
| `--by family` / `--by set` | one group per family or per test set | family |
| `--setting both` / `3task` / `5task` | which pretrained models to include | both |
| `--ylim -4 6` | y-range (dB); lower values are marked at the bottom edge | -4 6 |
| `--tta_label "..."` | legend name of the existing method | Existing AiO TTA |

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
