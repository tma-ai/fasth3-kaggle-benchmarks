import subprocess, shutil, json, urllib.request, os, platform
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e: return f"ERR {e}"
print("== nvidia-smi"); print(sh("nvidia-smi"))
print("== GPUs"); print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"))
print("== CUDA / torch")
try:
    import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
except Exception as e: print("torch ERR", e)
print("== RAM"); print(sh("free -g"))
print("== Disk"); [print(p, round(shutil.disk_usage(p).free/1e9,1), "GB free") for p in ["/kaggle/working", "/tmp"] if os.path.exists(p)]
print("== Internet")
try: print("HF reachable:", urllib.request.urlopen("https://huggingface.co", timeout=15).status)
except Exception as e: print("no internet:", e)
print("== Python", platform.python_version())
