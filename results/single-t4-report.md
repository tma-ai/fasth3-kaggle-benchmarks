# FastH3-Omni on free Kaggle 2× T4: results (2026-11-08)

**Verdict: usable for tests only.** FastH3-Omni ref2va loads and renders correct, good-looking video with sound on one T4. It is too slow for production: about 5 min for a 5 s clip at 512×288, and 24 min at 768×432.

## Kernel folders
- `tools/kaggle/h3-probe/` is a CPU-only probe, so it used no GPU quota. It installs ComfyUI master, writes the node schemas to `out/object_info_h3.json` and times one download.
- `tools/kaggle/fasth3-t4/` is the GPU test. The script is `fasth3_t4.py` and the results are in `out/`:
  - `metrics.json`
  - `comfy.log`
  - three MP4s
  - `frames/`

## Setup
- **ComfyUI:** master c76e64f (2026-11-08), headless on 127.0.0.1. There was no tunnel and no web UI.
- **Software:** torch 2.11 cu130 and comfy-kitchen 0.2.41. The kitchen CUDA backend was available.
- **Models:** all from `FastVideo/FastH3-Omni-Comfy`, which is **not gated**, so no token was used.
  - `fasth3_omniref_8step_pruned_int8_convrot` (22.2 GB)
  - text encoder `qwen3vl_32b_minimax_h3_nvfp4_awq` (15.7 GB)
  - video VAE int8 (2.8 GB)
  - audio VAE fp32 (0.6 GB)
- **Install and download:** install took 47 s and the 41 GB of downloads took 204 s.
- **Graph:** the same graph as the skill.
  - MiniMax H3 Reference to Video with one reference picture: a coffee-cup photo from Wikimedia, no people.
  - SigmaShift 12/3, then euler / simple, 8 steps, cfg 1.
  - The test character is an invented woman, described in the prompt only, saying the line "It's still warm."
- **Precision:** ComfyUI ran the diffusion model with **bf16 weights cast to fp32** (no bf16 on Turing), and the text encoder in fp16. There was no NaN and no black output.

## Results (all clips 124 frames = 5.17 s)

| Job | Size | Attention | Status | Wall time | s per s of video | Sampling |
|---|---|---|---|---|---|---|
| j1 | 512×288 (0.15 MP) | dense | OK | 432 s (cold) | 84 | 33.5 s/step |
| j2 | 512×288 | VSA keep 10 | OK | 297 s (warm) | 58 | 33.2 s/step |
| j3 | 768×432 (0.33 MP) | VSA | OK | 1,435 s (24 min) | 278 | 166 s/step |
| j4 | 1024×576 (0.59 MP) | VSA | **Interrupted at step 7 of 8** by my 60-min cap | > 3,617 s | > 700 | 473 s/step |

- **VSA did nothing at 512×288.** j1 and j2 have identical frame statistics, and the sequence is below the node's `min_tokens` of 12,288, so it stays dense. The j1/j2 gap is loading: about 135 s of model and text-encoder load on a cold start.
- **1024×576 does fit in memory.** It ran for an hour without running out of memory. A full render would take about 65–70 min per 5 s, so it is impractical.
- **Peak usage, the same for every job:**
  - GPU 0: 14,653 MiB of 15,360. ComfyUI's dynamic VRAM streams the 21.3 GB model from RAM.
  - GPU 1: 133 MiB, so it was never used.
  - System RAM: 27.2 GB of 31. ComfyUI's resident memory was 23.0 GB.
  - There was no swapping and no crash.
- **The largest size that fits** is 1024×576 at 5 s, but only 512×288 and 768×432 finish in a usable time.

## Output and quality
- **Clips:** `fasth3-t4/out/j1_512x288_dense.mp4`, `j2_512x288_vsa.mp4` and `j_768x432_vsa.mp4`.
- **Frames:** `fasth3-t4/out/frames/*_0N.png` and `*_strip.png`.
- **Picture, from the frames I viewed (strips and one full frame at 768):**
  - Photoreal and coherent, with a believable kitchen and window light.
  - The cup keeps the reference's white shape.
  - Real gravity: the cup is lifted, sipped and lowered.
  - Natural skin and a slow push-in.
  - 768×432 is noticeably sharper than 512×288.
  - Identity holds within each clip.
- **Sound:** the audio track is present in every clip (mean −24.5 dB, peak −5.5 dB) and the mouth moves at the end, which suggests the line was spoken. I did not run speech recognition or listen frame by frame, so speech accuracy and lip-sync are **unchecked**.

## Blockers
1. **Speed.**
   - **Compute:** the T4 has no bf16 and no fp8, so every layer is cast to fp32.
   - **Memory:** the 21 GB model does not fit in 15 GB, so the weights stream over PCIe on every step.
   - **Result:** about 33 s per step at 0.15 MP and 166 s per step at 0.33 MP. That is roughly 15–30× slower than the 4090 figures in the skill.
2. **The second T4 is idle.** One ComfyUI process uses only cuda:0. `MultiGPU_WorkUnits` exists, but at cfg 1.0 there is only one condition to split. I did not test it.
3. **RAM headroom is tight** (27 of 31 GB). Adding a LoRA, the SelfLift upscaler or more reference pictures and audio could push it over.
4. **Quota:** a 30 h weekly quota buys about 70–80 draft clips at 512×288, or about 1 clip an hour at 768×432 once loading is counted.

## GPU hours used
- The probe used **none**.
- fasth3-t4 v1 ran for about 101 min of wall time, which is **about 1.7 GPU-hours**. That is under the 4-hour cap.

## Recommendation
**Usable for tests only.** Use it for free prompt, staging and lip-sync drafts at 512×288, about 5 min per 5 s clip once warm. Batch several jobs into one kernel so models load once.

Final renders still need a 24–32 GB card with at least 64–96 GB of RAM (Vast or Novita, under the money rules).

Possible next tests:
- `MultiGPU_WorkUnits`
- putting the text encoder on GPU 1
- the SelfLift two-stage sampler (it samples most steps at low resolution)
- running ASR and lip-sync QC on these clips
