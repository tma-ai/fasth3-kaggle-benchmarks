# FastH3-Omni scene 1 (10 s, 244 frames) on Kaggle 2x T4 using BOTH GPUs as one VRAM pool.
# Method: pollockjj/ComfyUI-MultiGPU DisTorch2 - video model computes on cuda:0, its layers are split
# across cuda:0 and cuda:1 (cuda:1 = donor instead of CPU RAM). Headless ComfyUI on 127.0.0.1 only.
import subprocess, json, os, sys, time, threading, urllib.request, shutil, glob, traceback
T0 = time.time()
DEADLINE = T0 + 145 * 60          # hard wall-clock budget for this kernel (< 150 min)
W = "/kaggle/working"; C = "/tmp/ComfyUI"; M = f"{C}/models"
os.environ.pop("HF_TOKEN", None); os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
METRICS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}, "jobs": [], "notes": []}
def save(): json.dump(METRICS, open(f"{W}/metrics.json", "w"), indent=1)
def el(): return round(time.time() - T0, 1)
def sh(c, t=3600):
    t0 = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(f"[{el()}s] $ {c}\n  rc={r.returncode} {time.time()-t0:.0f}s\n{(r.stdout+r.stderr)[-3000:]}", flush=True); return r

# ---------- resource monitor (peaks + 5 s time series of VRAM and utilisation per GPU) ----------
import psutil
PEAK = {}; COMFY_PID = [None]; SERIES = []
def reset_peaks():
    PEAK.clear(); PEAK.update({"gpu0_mib": 0, "gpu1_mib": 0, "gpu0_util_max": 0, "gpu1_util_max": 0, "ram_used_gb": 0, "comfy_rss_gb": 0}); SERIES.clear()
reset_peaks(); GLOBAL = {"gpu0_mib": 0, "gpu1_mib": 0, "ram_used_gb": 0, "comfy_rss_gb": 0}
def mon():
    n = 0
    while True:
        try:
            out = subprocess.run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout
            row = {"t": el()}
            for line in out.strip().splitlines():
                i, u, ut = [x.strip() for x in line.split(",")]; k = f"gpu{i}_mib"
                PEAK[k] = max(PEAK.get(k, 0), int(u)); GLOBAL[k] = max(GLOBAL.get(k, 0), int(u))
                PEAK[f"gpu{i}_util_max"] = max(PEAK.get(f"gpu{i}_util_max", 0), int(ut))
                row[f"g{i}"] = int(u); row[f"u{i}"] = int(ut)
            vm = psutil.virtual_memory(); used = round((vm.total - vm.available) / 2**30, 2)
            PEAK["ram_used_gb"] = max(PEAK["ram_used_gb"], used); GLOBAL["ram_used_gb"] = max(GLOBAL["ram_used_gb"], used)
            row["ram"] = used
            if COMFY_PID[0]:
                try:
                    rss = round(psutil.Process(COMFY_PID[0]).memory_info().rss / 2**30, 2)
                    PEAK["comfy_rss_gb"] = max(PEAK["comfy_rss_gb"], rss); GLOBAL["comfy_rss_gb"] = max(GLOBAL["comfy_rss_gb"], rss)
                except Exception: pass
            if n % 5 == 0: SERIES.append(row)
            n += 1
        except Exception: pass
        time.sleep(1)
threading.Thread(target=mon, daemon=True).start()

# ---------- install: ComfyUI master + pollockjj/ComfyUI-MultiGPU (default branch main) ----------
t = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nvidia-smi topo -m; python -V; pip list 2>/dev/null | grep -i -E '^(torch|triton|torchaudio|torchvision) '")
sh(f"git clone -q --depth 1 https://github.com/Comfy-Org/ComfyUI.git {C} && cd {C} && git log -1 --format='%H %cd'")
sh(f"git clone -q --depth 1 --branch main https://github.com/pollockjj/ComfyUI-MultiGPU.git {C}/custom_nodes/ComfyUI-MultiGPU && cd {C}/custom_nodes/ComfyUI-MultiGPU && git log -1 --format='%H %cd'")
import torch
open("/tmp/cons.txt", "w").write(f"torch=={torch.__version__}\n")
sh(f"cd {C} && pip install -q -c /tmp/cons.txt -r requirements.txt hf_xet")
if os.path.exists(f"{C}/custom_nodes/ComfyUI-MultiGPU/requirements.txt"):
    sh(f"cd {C}/custom_nodes/ComfyUI-MultiGPU && pip install -q -c /tmp/cons.txt -r requirements.txt")
sh("python -c \"import torch; print('p2p 0->1', torch.cuda.can_device_access_peer(0,1), '1->0', torch.cuda.can_device_access_peer(1,0))\"")
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

os.makedirs(f"{C}/input", exist_ok=True)
for f in ("marcus.png", "room.png"):
    hits = glob.glob(f"/kaggle/input/**/{f}", recursive=True)
    print("ref", f, hits, flush=True)
    shutil.copy(hits[0], f"{C}/input/{f}")

# ---------- ComfyUI server ----------
LOG = open(f"{W}/comfy.log", "a")
PROC = [None]; ARGS = [None]
def start(extra=()):
    stop()
    LOG.write(f"\n===== start {extra} at {el()}s =====\n"); LOG.flush()
    PROC[0] = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", *extra], cwd=C, stdout=LOG, stderr=subprocess.STDOUT)
    COMFY_PID[0] = PROC[0].pid; ARGS[0] = tuple(extra)
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
    COMFY_PID[0] = None; time.sleep(3)
def api(path, data=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:8188{path}", data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

OI = {}
def load_object_info():
    oi = api("/object_info", timeout=180); OI.clear(); OI.update(oi)
    keys = [k for k in oi if "distorch" in k.lower() or "multigpu" in k.lower() or "nodes_multigpu" in str(oi[k].get("python_module", ""))]
    keys += ["UNETLoader", "CLIPLoader", "VAELoader"]
    sel = {k: {"input": oi[k].get("input"), "module": oi[k].get("python_module")} for k in keys if k in oi}
    json.dump(sel, open(f"{W}/object_info_multigpu.json", "w"), indent=1)
    for k in ("UNETLoaderDisTorch2MultiGPU", "CLIPLoaderMultiGPU", "VAELoaderMultiGPU", "SelectModelDevice", "SelectCLIPDevice", "SelectVAEDevice", "MultiGPU_WorkUnits"):
        if k in oi: print("OBJECT_INFO", k, json.dumps(oi[k].get("input"))[:2500], flush=True)
        else: print("OBJECT_INFO", k, "MISSING", flush=True)
    METRICS["multigpu_nodes"] = [k for k in keys if k in oi]; save()

def opts(node, inp):
    d = OI[node]["input"]
    spec = d.get("required", {}).get(inp) or d.get("optional", {}).get(inp)
    if spec is None: raise KeyError(f"{node} has no input {inp}")
    if isinstance(spec[0], list): return spec[0]
    if spec[0] == "COMBO": return spec[1].get("options", [])
    return None
def dev(node, inp, want):
    o = opts(node, inp)
    if o is None or want in o: return want
    m = [x for x in o if x.endswith(want.split(":")[-1]) and "cuda" in x]
    if m: return m[0]
    raise ValueError(f"{node}.{inp}: {want} not in {o}")
def check(g):
    # every input must exist in the node's real schema
    for nid, n in g.items():
        ct = n["class_type"]
        if ct not in OI: raise KeyError(f"node {ct} not available")
        d = OI[ct]["input"]; names = set(d.get("required", {})) | set(d.get("optional", {}))
        for k in n["inputs"]:
            base = k.split(".")[0]
            if base not in names and k not in names: raise KeyError(f"{ct} has no input {k} (has {sorted(names)})")

PROMPT = """subject_definitions: <Subject 1> is MARCUS VANCE, the man in <Picture 1>: an older tycoon with thick silver-grey hair swept back, a weathered tanned face, navy pinstripe double-breasted suit, burgundy tie, gold signet ring, gold watch. <Subject 2> is ELENA ROSTOVA, the woman seated on the right in <Picture 2>: dark hair in a low bun, charcoal suit, silver necklace. The room is the boardroom in <Picture 2>: a long polished walnut table, brass dome pendant lamps, glass cases of model ships, a floor-to-ceiling window with a rainy night harbour and a ship under work lights.
summary: [reference generation] A {secs}-second scene in a shipping company boardroom at night: <Subject 1> stamps a foreclosure deed and announces his victory while <Subject 2> watches.
retention_analysis: <Subject 1> (all shots): fully_preserved - face, hair, age and suit identical to <Picture 1>. <Subject 2> (shot 1): fully_preserved - identical to <Picture 2>. The room (all shots): fully_preserved - same table, lamps, model cases and window as <Picture 2>.
detailed_description: [Shot 1] 0 to 4 seconds. Wide shot across the long walnut table toward the rain-streaked window, matching <Picture 2>: <Subject 1> sits at the left near the window end and brings a brass seal down onto the deed with one solid thud; <Subject 2> sits on the right side and watches him, silent. <Subject 1> (S1), the older man, in a deep gravelly self-satisfied voice, says: <d>[English] Pier Nine is officially foreclosed.</d> [Shot 2] 4 to 10 seconds. Hard cut to a medium close-up of <Subject 1> in profile in his leather chair, the rainy harbour window behind him; he leans back with a smug half-smile and says: <d>[English] Eighty million in salvage rights, transferred to Vance Holdings.</d> Then three heavy knocks sound on the doors off-screen; his smile drops and his eyes turn sharply toward the sound. Real gravity, real-time motion. Real camera, real photographic texture, not CGI, natural skin with pores, film grain. Exactly two people. Rain only outside the glass. No readable text, no subtitles.
overall_soundscape: rain on the window, a low city hum, one wooden seal thud, three heavy knocks on a wooden door.
non_diegetic_music: N/A"""

UNET = "fasth3_omniref_8step_pruned_int8_convrot.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
def graph(cfg, w, h, length, seed=7, prefix="h3"):
    g = {
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "LoadImage", "inputs": {"image": "marcus.png"}},
        "31": {"class_type": "LoadImage", "inputs": {"image": "room.png"}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["3", 0], "prompt": PROMPT.format(secs=round(length / 24, 1)), "width": w, "height": h, "length": length,
              "ref_image_size": "match", "vae": ["4", 0], "audio_vae": ["5", 0], "ref_images.ref_image_0": ["30", 0], "ref_images.ref_image_1": ["31", 0]}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "21": {"class_type": "BlockSparseAttention", "inputs": {"model": ["2", 0], "selection": "vsa", "selection.keep_percent": 10.0, "start_percent": 0.0, "end_percent": 1.0,
               "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": True}},
        "7": {"class_type": "BasicGuider", "inputs": {"model": ["21", 0], "conditioning": ["6", 0]}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["7", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 24.0, "audio": ["13", 0]}},
        "15": {"class_type": "SaveVideo", "inputs": {"video": ["14", 0], "filename_prefix": f"2g/{prefix}", "format": "auto", "format.codec": "auto"}},
    }
    if cfg["kind"] == "distorch":
        N = "UNETLoaderDisTorch2MultiGPU"
        g["1"] = {"class_type": N, "inputs": {"unet_name": UNET, "weight_dtype": "default",
                  "compute_device": dev(N, "compute_device", "cuda:0"), "donor_device": dev(N, "donor_device", "cuda:1"),
                  "virtual_vram_gb": cfg["virtual_vram_gb"], "expert_mode_allocations": cfg["alloc"], "eject_models": True}}
    else:   # (a) encoder-only split: video model on cuda:0 (default), TE + VAEs on cuda:1 via core Select*Device nodes
        g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}}
        g["40"] = {"class_type": "SelectCLIPDevice", "inputs": {"clip": ["3", 0], "device": dev("SelectCLIPDevice", "device", "cuda:1")}}
        g["41"] = {"class_type": "SelectVAEDevice", "inputs": {"vae": ["4", 0], "device": dev("SelectVAEDevice", "device", "cuda:1")}}
        g["42"] = {"class_type": "SelectVAEDevice", "inputs": {"vae": ["5", 0], "device": dev("SelectVAEDevice", "device", "cuda:1")}}
        g["6"]["inputs"].update({"clip": ["40", 0], "vae": ["41", 0], "audio_vae": ["42", 0]})
        g["12"]["inputs"]["vae"] = ["41", 0]; g["13"]["inputs"]["vae"] = ["42", 0]
    check(g)
    return g

def run_job(name, cfg, w, h, length, timeout_s, split_check=False):
    # split_check: once sampling has run 150 s, cuda:1 must hold >= 6 GB (model weights), else abort early (status no_split)
    reset_peaks()
    j = {"name": name, "cfg": cfg, "server_args": list(ARGS[0] or ()), "w": w, "h": h, "frames": length, "video_s": round(length / 24, 3), "mp": round(w * h / 1e6, 3), "start_s": el()}
    print(f"\n######## JOB {j}", flush=True)
    log_off = os.path.getsize(f"{W}/comfy.log")
    t1 = time.time()
    try:
        if PROC[0] is None or PROC[0].poll() is not None: raise RuntimeError("comfy not running")
        g = graph(cfg, w, h, length, prefix=name)
        j["graph_node1"] = g["1"]; print("GRAPH node 1:", json.dumps(g["1"]), flush=True)
        r = api("/prompt", {"prompt": g})
        pid = r["prompt_id"]; j["prompt_id"] = pid
        status = None; samp_t = None; last_print = 0
        while True:
            if PROC[0].poll() is not None: status = "comfy_died"; break
            if samp_t is None or time.time() - last_print > 120:
                try:
                    with open(f"{W}/comfy.log", errors="ignore") as fh: fh.seek(log_off); new = fh.read()
                except Exception: new = ""
                if samp_t is None and ("Model Initializing" in new or "/8 [" in new):
                    samp_t = time.time(); j["sampling_start_s"] = round(samp_t - t1, 1); print(f"[{el()}s] sampling started", flush=True)
                if time.time() - last_print > 120:
                    last_print = time.time(); prog = new.replace("\r", "\n").strip().splitlines()[-1:] if new else []
                    print(f"[{el()}s] {name} peaks={dict(PEAK)} last={prog}", flush=True)
            if split_check and samp_t and time.time() - samp_t > 150 and PEAK.get("gpu1_mib", 0) < 6000:
                try: api("/interrupt", {})
                except Exception: pass
                status = "no_split"; time.sleep(15); break
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
    j["peaks"] = dict(PEAK); j["series"] = list(SERIES)
    j["s_per_video_s"] = round(j["wall_s"] / j["video_s"], 1)
    txt = open(f"{W}/comfy.log", errors="ignore").read()[log_off:]
    keep = [l for l in txt.replace("\r", "\n").splitlines() if any(s in l for s in ("loaded", "Loaded", "Requested", "dtype", "it/s", "s/it", "Prompt executed", "Error", "error", "OutOfMemory", "out of memory", "Killed", "sparse", "VSA", "offload", "lowvram", "Model ", "DisTorch", "MultiGPU", "donor", "recip", "alloc", "cuda:1", "Allocation"))]
    j["log"] = [l[-300:] for l in keep][-120:]
    j["log_tail"] = txt.splitlines()[-25:]
    vids = sorted(glob.glob(f"{C}/output/2g/{name}*.mp4"))
    if vids:
        v = vids[-1]; dst = f"{W}/{name}.mp4"; shutil.copy(v, dst); j["mp4"] = os.path.basename(dst)
        os.makedirs(f"{W}/frames", exist_ok=True)
        sh(f"ffmpeg -loglevel error -y -i {dst} -vf \"select='not(mod(n\\,30))'\" -vsync vfr {W}/frames/{name}_%02d.png")
        sh(f"ffmpeg -loglevel error -y -i {dst} -vf \"fps=1,scale=240:-1,tile=5x2\" -frames:v 1 {W}/frames/{name}_strip.png")
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

def split_used(j):  # cuda:1 really held model weights while sampling (TE is on cuda:0 in DisTorch configs)
    return j["status"] == "success" and j["peaks"].get("gpu1_mib", 0) >= 6000
def oom(j): return "out of memory" in json.dumps(j.get("messages", "")).lower() + json.dumps(j.get("log_tail", "")).lower() or j["status"] == "comfy_died"

# DisTorch2 expert allocation (DisTorch "bytes" mode, 1024-based): 5 GB of the ~19.9 GiB video model stays on cuda:0
# (the compute card keeps ~9 GB free for 480p x 244-frame activations), 13 GB lives on cuda:1 (donor), the last ~2 GB in CPU RAM.
CFG = {"kind": "distorch", "alloc": "cuda:0,5gb;cuda:1,13gb;cpu,*", "virtual_vram_gb": 15.0}
CFG_LOWACT = {"kind": "distorch", "alloc": "cuda:0,3gb;cuda:1,13gb;cpu,*", "virtual_vram_gb": 17.0}
SIZES = [(480, 848), (480, 864)]   # 480p vertical: 848 is the nearest multiple of 16 to 854; 864 if 848 is rejected
TARGET = 480
def rem(): return DEADLINE - time.time()
def render(tag, cfg, check):
    for (w, h) in SIZES:
        j = run_job(f"{tag}_{w}x{h}", cfg, w, h, 244, min(rem() - 180, 115 * 60), split_check=check)
        if not str(j["status"]).startswith("exception: HTTP Error 400"): return j
        METRICS["notes"].append(f"{w}x{h} rejected by validation: {j['status']}")
    return j
def failed(j): return j["status"] != "success"
try:
    METRICS["phases"]["setup_total_s"] = el()
    # 1) DisTorch2 split, ComfyUI default (DynamicVRAM) loader
    start(); load_object_info(); METRICS["phases"]["comfy_ready_s"] = el(); save()
    j = render("dyn", CFG, True)
    if failed(j) and oom(j) and j["status"] != "no_split" and rem() > 60 * 60:
        METRICS["notes"].append("dyn OOM -> fewer weights on cuda:0")
        start(); j = render("dyn_lowact", CFG_LOWACT, True)
    if j["status"] == "no_split" or (failed(j) and not j["status"].startswith("timeout")):
        METRICS["notes"].append(f"dyn: status={j['status']} gpu1 peak={j['peaks'].get('gpu1_mib')} MiB -> legacy loader (--disable-dynamic-vram) so DisTorch placement applies")
        if rem() > 50 * 60:
            start(("--disable-dynamic-vram", "--cache-none"))
            j = render("legacy", CFG, True)
            if failed(j) and oom(j) and j["status"] != "no_split" and rem() > 50 * 60:
                start(("--disable-dynamic-vram", "--cache-none")); j = render("legacy_lowact", CFG_LOWACT, True)
    if failed(j) and rem() > 55 * 60:
        METRICS["notes"].append("no working 2-GPU split; rendering 480p with DisTorch loader and no split check so a video is delivered")
        start(); j = render("fallback", CFG, False)
    METRICS["final_job"] = j["name"]
except Exception as e:
    METRICS["notes"].append(f"fatal: {e}"); METRICS["trace"] = traceback.format_exc()
finally:
    stop(); METRICS["global_peaks"] = dict(GLOBAL); METRICS["elapsed_s"] = el(); save()
    print(json.dumps({k: v for k, v in METRICS.items() if k not in ("system_stats",)}, default=str)[:6000], flush=True)
