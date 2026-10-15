Released source checkpoints (not tracked by git). `scripts/run.sh` expects this layout; edit the `CK` table at
its top if your file names differ (a missing file skips that backbone/setting).

```
pretrain/model.ckpt, pretrain/epoch=80.ckpt                       PromptIR 3-task / 5-task
pretrain/adair/adair3d.ckpt, adair5d.ckpt                          AdaIR       github.com/c-yn/AdaIR
pretrain/bioir/BioIR3D.ckpt, BioIR5D.ckpt                          BioIR       github.com/c-yn/BioIR (All-in-one)
pretrain/moceir/MoCE_IR_AIO3/last.ckpt, MoCE_IR_AIO5/last.ckpt      MoCE-IR     github.com/eduardzamfir/MoCE-IR
pretrain/mirage/3Deg_small.ckpt, 5Deg_small.ckpt                   MIRAGE      github.com/Amazingren/MIRAGE
pretrain/r2r/train_ckpt_3D_finetune/last.ckpt + save_prompts_3D/last/*.pt     R2R 3-task (+ degradation bank)
pretrain/r2r/train_ckpt_5D_finetune/last.ckpt + save_prompts_5D/last/*.pt     R2R 5-task (+ degradation bank)
```

Lightning checkpoints (`net.`-prefixed keys) load as-is; the variant (3/5-task, tiny/small, MoCE-IR vs -S) is read
off the checkpoint. Check them with `python tools/check_backbones.py <name>=<ckpt> ...`.
The RDDM generator of DCTTA has no checkpoint: it is trained from scratch on the target set.
