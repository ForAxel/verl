import os, torch
import torch.multiprocessing as mp

def worker(q):
    print(f"[Worker] PID={os.getpid()} visible={os.environ.get('MUSA_VISIBLE_DEVICES')}")
    print("count=", torch.musa.device_count(), "cur=", torch.musa.current_device(), flush=True)
    for _ in range(2):
        x = q.get()
    print("[Worker] got", x.device, "nan=", torch.isnan(x).any().item(), "sum=", x.float().sum().item(), flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["MUSA_VISIBLE_DEVICES"] = "6,7"

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    
    import time
    time.sleep(2)

    for dev in ["musa:0", "musa:1"]:
        with torch.musa.device(dev):
            t = torch.randn(1024, device=dev, dtype=torch.float32)
            print("[Main] put", t.device, flush=True)
            q.put(t)
    p.join()