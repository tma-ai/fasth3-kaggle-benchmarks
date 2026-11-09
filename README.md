# FastH3 on free Kaggle T4s: benchmarks

Can an open reference-to-video model with native audio and lip-sync (MiniMax H3 / FastH3-Omni, run headless in ComfyUI) work on Kaggle's **free 2× Tesla T4 (15 GB each)**? I wanted to know before paying for GPUs to render my AI micro-dramas.

## Results so far (single T4, 5.2 s clips, 124 frames)

| Resolution | Wall time | Compute per second of video | Peak VRAM (GPU 0) | Peak RAM |
|---|---|---|---|---|
| 512×288, cold start | 432 s | 84 s | 14.65 GB | 27 GB |
| 512×288, warm | 297 s | 58 s | 14.65 GB | 26 GB |
| 768×432 | 1435 s (24 min) | 278 s | 14.65 GB | 27 GB |
| 1024×576 | stopped at 60 min | over 700 s | 14.65 GB | 27 GB |

- **Setup:** about 47 s to install ComfyUI, and about 3.5 min to download 41 GB of models (no token needed).
- **Quality:** the picture is photoreal and coherent, the motion follows real gravity, and the speech is intelligible. The test line was "It's still warm.", transcribed correctly.
- **Why it's slow:** the T4 has no bf16, and the 22 GB int8 model streams from RAM on every step. GPU 1 sits idle.

![768x432 strip](results/frames/j_768x432_vsa_strip.png)

## Dual-T4 attempt

Splitting the model across both cards with a DisTorch-style loader did put real load on both GPUs (14.9 GB and 13.4 GB). But a full 10-second clip at 480×848 (244 frames) ran out of memory on GPU 0 during sampling. The next test renders the same scene as short 2.5–4 s shots, which cuts the activation memory.

## Verdict (for now)

Free T4s work for **drafts**: 512×288 at about 5 min per 5 s clip is enough to test prompts, staging and lip-sync. Final HD renders need a 24 GB+ card. The next step is a single 192 GB AMD Instinct MI300X on ROCm, where the full-precision model fits on one card.

## Files
- `kernels/`: the Kaggle kernel scripts (headless ComfyUI on localhost; resource monitor; per-job metrics).
- `results/`: the report, frame strips and a sample clip. Every person in the test clips is invented.

— David Udochukwu ([@tma-ai](https://github.com/tma-ai))
