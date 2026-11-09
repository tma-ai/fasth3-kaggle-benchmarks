# FastH3-Omni (ref2va) fit + speed test on Kaggle 2x T4. Headless ComfyUI on 127.0.0.1 only.
# Writes metrics.json, comfy.log, MP4s and PNG frames to /kaggle/working.
import subprocess, json, os, sys, time, threading, urllib.request, shutil, glob, traceback
T0 = time.time()
DEADLINE = T0 + 150 * 60          # hard wall-clock budget for this kernel
W = "/kaggle/working"; C = "/tmp/ComfyUI"; M = f"{C}/models"
os.environ.pop("HF_TOKEN", None); os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
METRICS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}, "jobs": [], "notes": []}
def save(): json.dump(METRICS, open(f"{W}/metrics.json", "w"), indent=1)
def el(): return round(time.time() - T0, 1)
def sh(c, t=3600):
    t0 = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(f"[{el()}s] $ {c}\n  rc={r.returncode} {time.time()-t0:.0f}s\n{(r.stdout+r.stderr)[-3000:]}", flush=True); return r

# ---------- resource monitor ----------
import psutil
PEAK = {}; COMFY_PID = [None]
def reset_peaks(): PEAK.clear(); PEAK.update({"gpu0_mib": 0, "gpu1_mib": 0, "ram_used_gb": 0, "comfy_rss_gb": 0})
reset_peaks(); GLOBAL = {"gpu0_mib": 0, "gpu1_mib": 0, "ram_used_gb": 0, "comfy_rss_gb": 0}
def mon():
    while True:
        try:
            out = subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout
            for line in out.strip().splitlines():
                i, u = [x.strip() for x in line.split(",")]; k = f"gpu{i}_mib"
                PEAK[k] = max(PEAK.get(k, 0), int(u)); GLOBAL[k] = max(GLOBAL.get(k, 0), int(u))
            vm = psutil.virtual_memory(); used = round((vm.total - vm.available) / 2**30, 2)
            PEAK["ram_used_gb"] = max(PEAK["ram_used_gb"], used); GLOBAL["ram_used_gb"] = max(GLOBAL["ram_used_gb"], used)
            if COMFY_PID[0]:
                try:
                    rss = round(psutil.Process(COMFY_PID[0]).memory_info().rss / 2**30, 2)
                    PEAK["comfy_rss_gb"] = max(PEAK["comfy_rss_gb"], rss); GLOBAL["comfy_rss_gb"] = max(GLOBAL["comfy_rss_gb"], rss)
                except Exception: pass
        except Exception: pass
        time.sleep(1)
threading.Thread(target=mon, daemon=True).start()

# ---------- install ----------
t = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python -V; pip list 2>/dev/null | grep -i -E '^(torch|triton|torchaudio|torchvision) '")
sh(f"git clone -q --depth 1 https://github.com/Comfy-Org/ComfyUI.git {C} && cd {C} && git log -1 --format='%H %cd'")
import torch
open("/tmp/cons.txt", "w").write(f"torch=={torch.__version__}\n")
sh(f"cd {C} && pip install -q -c /tmp/cons.txt -r requirements.txt")
sh("python -c \"import comfy_kitchen as ck; b=ck.list_backends(); print({k:(v['available'],v['unavailable_reason']) for k,v in b.items()})\"")
METRICS["phases"]["install_s"] = round(time.time() - t, 1); METRICS["torch"] = torch.__version__; save()

# ---------- downloads (no token) ----------
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
OMNI = "FastVideo/FastH3-Omni-Comfy"
FILES = [(OMNI, "diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors"),
         (OMNI, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
         (OMNI, "vae/minimax_h3_video_vae_int8_convrot.safetensors"),
         (OMNI, "vae/minimax_h3_audio_vae_fp32.safetensors")]
def dl(f):
    t1 = time.time(); p = hf_hub_download(f[0], f[1], local_dir=M, token=False)
    return f[1], round(os.path.getsize(p) / 1e9, 2), round(time.time() - t1, 1)
t = time.time()
with ThreadPoolExecutor(4) as ex: res = list(ex.map(dl, FILES))
METRICS["downloads"] = res; METRICS["phases"]["download_s"] = round(time.time() - t, 1); save()
print("downloads:", res, flush=True)

# reference picture: a neutral object photo (Wikimedia Commons), no people
os.makedirs(f"{C}/input", exist_ok=True)
try:
    req = urllib.request.Request("https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG", headers={"User-Agent": "Mozilla/5.0 kaggle-h3-test"})
    open(f"{C}/input/cup.jpg", "wb").write(urllib.request.urlopen(req, timeout=60).read())
except Exception as e:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1024, 768), (190, 160, 120)); d = ImageDraw.Draw(im)
    d.ellipse((380, 300, 640, 420), fill=(245, 245, 240)); im.save(f"{C}/input/cup.jpg"); METRICS["notes"].append(f"cup download failed, drew one: {e}")

# ---------- ComfyUI server ----------
LOG = open(f"{W}/comfy.log", "a")
PROC = [None]
def start(extra=()):
    stop()
    LOG.write(f"\n===== start {extra} at {el()}s =====\n"); LOG.flush()
    PROC[0] = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", *extra], cwd=C, stdout=LOG, stderr=subprocess.STDOUT)
    COMFY_PID[0] = PROC[0].pid
    for _ in range(300):
        if PROC[0].poll() is not None: raise RuntimeError("comfy exited at start")
        try: urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3); return
        except Exception: time.sleep(2)
    raise RuntimeError("comfy did not start")
def stop():
    if PROC[0] and PROC[0].poll() is None:
        PROC[0].terminate()
        try: PROC[0].wait(30)
        except Exception: PROC[0].kill()
    COMFY_PID[0] = None
def api(path, data=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:8188{path}", data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

PROMPT = """subject_definitions: <Subject 1> is a woman in her early thirties with shoulder-length dark brown hair, clear even skin with no marks, wearing a thick grey knit sweater. <Subject 2> is the white ceramic cup and saucer in <Picture 1>.
summary: [reference generation] A {secs}-second single shot in a quiet kitchen in the morning: <Subject 1> lifts <Subject 2> from the wooden table, takes a small sip and speaks one short line.
retention_analysis: <Subject 2> (appears in [Shot 1]): fully_preserved - the cup's shape, white glaze, coffee colour and saucer stay identical; its position and the framing follow this shot, not <Picture 1>.
detailed_description: [Shot 1] Medium close-up at eye level, the camera slowly pushes in from a waist-level start; soft window light from the left, the kitchen softly out of focus behind her. <Subject 1> sits at the table, picks up <Subject 2> with her right hand, the saucer stays on the table, she takes a small sip, lowers the cup a little and smiles. <Subject 1> (S1), the woman on-screen with a warm, calm voice, says: <d>[English] It's still warm.</d> Real gravity, the cup has weight. Real camera, real photographic texture, not CGI, natural skin with pores and fine lines, no beauty filter, film grain. No extra people, no wardrobe changes, no readable text, no subtitles.
overall_soundscape: quiet kitchen room tone, a soft clink of porcelain as the cup leaves the saucer, faint birdsong outside the window.
non_diegetic_music: N/A"""

def graph(w, h, length, te, attn, seed=7, prefix="h3"):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "fasth3_omniref_8step_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "LoadImage", "inputs": {"image": "cup.jpg"}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["3", 0], "prompt": PROMPT.format(secs=round(length / 24, 1)), "width": w, "height": h, "length": length,
              "ref_image_size": "match", "vae": ["4", 0], "audio_vae": ["5", 0], "ref_images.ref_image_0": ["30", 0]}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "7": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["6", 0]}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["7", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 24.0, "audio": ["13", 0]}},
        "15": {"class_type": "SaveVideo", "inputs": {"video": ["14", 0], "filename_prefix": f"t4/{prefix}", "format": "auto", "format.codec": "auto"}},
    }
    if attn == "vsa":
        g["21"] = {"class_type": "BlockSparseAttention", "inputs": {"model": ["2", 0], "selection": "vsa", "selection.keep_percent": 10.0, "start_percent": 0.0, "end_percent": 1.0,
                   "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": True}}
        g["7"]["inputs"]["model"] = ["21", 0]
    return g

def run_job(name, w, h, length, te, attn, timeout_s):
    reset_peaks()
    j = {"name": name, "w": w, "h": h, "frames": length, "video_s": round(length / 24, 3), "mp": round(w * h / 1e6, 3), "te": te, "attn": attn, "start_s": el()}
    print(f"\n######## JOB {j}", flush=True)
    log_off = os.path.getsize(f"{W}/comfy.log")
    t1 = time.time()
    try:
        if PROC[0] is None or PROC[0].poll() is not None: start()
        r = api("/prompt", {"prompt": graph(w, h, length, te, attn, prefix=name)})
        pid = r["prompt_id"]; j["prompt_id"] = pid
        status = None
        while True:
            if PROC[0].poll() is not None: status = "comfy_died"; break
            if time.time() - t1 > timeout_s or time.time() > DEADLINE - 120:
                try: api("/interrupt", {})
                except Exception: pass
                status = "timeout_interrupted"; time.sleep(15); break
            try: hst = api(f"/history/{pid}")
            except Exception: hst = {}
            if pid in hst:
                st = hst[pid].get("status", {}); status = st.get("status_str", "?")
                j["messages"] = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
                break
            time.sleep(3)
    except Exception as e:
        status = f"exception: {e}"; j["trace"] = traceback.format_exc()[-1500:]
    j["status"] = status; j["wall_s"] = round(time.time() - t1, 1)
    j["peaks"] = dict(PEAK)
    j["s_per_video_s"] = round(j["wall_s"] / j["video_s"], 1)
    # log excerpt (loading / sampling lines)
    txt = open(f"{W}/comfy.log", errors="ignore").read()[log_off:]
    keep = [l for l in txt.splitlines() if any(s in l for s in ("loaded", "Loaded", "Requested", "dtype", "it/s", "s/it", "Prompt executed", "Error", "error", "OutOfMemory", "out of memory", "Killed", "sparse", "VSA", "vsa", "offload", "lowvram", "Model ", "weight"))]
    j["log"] = [l[-300:] for l in keep][-80:]
    j["log_tail"] = txt.splitlines()[-25:]
    # outputs
    vids = sorted(glob.glob(f"{C}/output/t4/{name}*.mp4"))
    if vids:
        v = vids[-1]; dst = f"{W}/{name}.mp4"; shutil.copy(v, dst); j["mp4"] = os.path.basename(dst)
        os.makedirs(f"{W}/frames", exist_ok=True)
        sh(f"ffmpeg -loglevel error -y -i {dst} -vf \"select='not(mod(n\\,30))'\" -vsync vfr {W}/frames/{name}_%02d.png")
        sh(f"ffmpeg -loglevel error -y -i {dst} -vf \"fps=2,scale=320:-1,tile=5x2\" -frames:v 1 {W}/frames/{name}_strip.png")
        sh(f"ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames,duration -of compact {dst}")
        try:
            from PIL import Image, ImageStat
            fr = sorted(glob.glob(f"{W}/frames/{name}_[0-9]*.png"))
            st = [ImageStat.Stat(Image.open(f).convert("L")) for f in fr]
            j["frame_mean_std"] = [(round(s.mean[0], 1), round(s.stddev[0], 1)) for s in st]
        except Exception as e: j["frame_stat_err"] = str(e)
        r = sh(f"ffmpeg -hide_banner -i {dst} -af volumedetect -f null - 2>&1 | grep -E 'mean_volume|max_volume'")
        j["audio_volume"] = r.stdout.strip()
    METRICS["jobs"].append(j); METRICS["global_peaks"] = dict(GLOBAL); METRICS["elapsed_s"] = el(); save()
    print(f"######## RESULT {name}: {status} wall={j['wall_s']}s peaks={j['peaks']}", flush=True)
    return j

TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
try:
    start()
    ss = api("/system_stats"); METRICS["system_stats"] = ss; save()
    j1 = run_job("j1_512x288_dense", 512, 288, 124, TE, "dense", 45 * 60)
    if j1["status"] != "success":
        METRICS["notes"].append("j1 failed with nvfp4 TE; retrying with --cache-none after restart")
        start(["--cache-none"])
        j1 = run_job("j1b_512x288_dense_cachenone", 512, 288, 124, TE, "dense", 45 * 60)
    if j1["status"] == "success":
        best_attn = "dense"
        rem = DEADLINE - time.time()
        if rem > j1["wall_s"] * 1.3 + 600:
            j2 = run_job("j2_512x288_vsa", 512, 288, 124, TE, "vsa", 30 * 60)
            if j2["status"] == "success" and j2["wall_s"] < j1["wall_s"]: best_attn = "vsa"
        last = METRICS["jobs"][-1] if METRICS["jobs"][-1]["status"] == "success" else j1
        for (w, h) in [(768, 432), (1024, 576)]:
            est = last["wall_s"] * (w * h / (last["w"] * last["h"])) ** 1.4
            rem = DEADLINE - time.time()
            print(f"next {w}x{h}: est {est:.0f}s, remaining {rem:.0f}s", flush=True)
            if est > rem - 300: METRICS["notes"].append(f"skipped {w}x{h}: est {est:.0f}s > remaining {rem:.0f}s"); break
            jj = run_job(f"j_{w}x{h}_{best_attn}", w, h, 124, TE, best_attn, min(rem - 180, 60 * 60))
            if jj["status"] != "success": break
            last = jj
except Exception as e:
    METRICS["notes"].append(f"fatal: {e}"); METRICS["trace"] = traceback.format_exc()
finally:
    stop(); METRICS["global_peaks"] = dict(GLOBAL); METRICS["elapsed_s"] = el(); save()
    print(json.dumps({k: v for k, v in METRICS.items() if k != "system_stats"}, indent=1)[:20000], flush=True)
