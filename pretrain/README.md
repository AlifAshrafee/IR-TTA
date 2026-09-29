Place the PromptIR source checkpoints here:

```
pretrain/
├── model.ckpt      # PromptIR, 3-task source (denoise sigma 15/25/50, Rain100L, RESIDE-OTS)
└── epoch=80.ckpt   # PromptIR, 5-task source (+ GoPro deblur, LOL low-light)
```

Both are PyTorch-Lightning checkpoints whose `state_dict` keys carry a `net.` prefix;
`dctta.utils.load_promptir_checkpoint` strips it.

`wavelet.mat` is no longer needed: the only band it provided (2x2 Haar low-pass) is
computed analytically in `dctta/losses.py::HaarLowPass`.

There is no checkpoint for the RDDM degradation generator -- it is trained from scratch
on the target set during adaptation.
