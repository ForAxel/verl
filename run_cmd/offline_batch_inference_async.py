"""
Usage:
python offline_batch_inference_async.py --model-path Qwen/Qwen2-VL-7B-Instruct

Note:
This demo shows the usage of async generation,
which is useful to implement an online-like generation with batched inference.
"""

import argparse
import asyncio
import dataclasses
import time

import sglang as sgl
from sglang.srt.server_args import ServerArgs

import torch
def snapshot(tag):
    allocated = torch.musa.memory_allocated()
    reserved  = torch.musa.memory_reserved()
    free,total = torch.musa.mem_get_info()
    used = total - free
    print('---'*20)
    print(f"{tag:20s}  |  allocated={allocated/1024**3:.2f} GB  "
          f"reserved={reserved/1024**3:.2f} GB  |  driver-used={used/1024**3:.2f} GB  "
          f"driver-total={total/1024**3:.2f} GB")
    print('---'*20)

class InferenceEngine:
    def __init__(self, **kwargs):
        self.engine = sgl.Engine(**kwargs)
        

    async def generate(self, prompt, sampling_params):
        result = await self.engine.async_generate(prompt, sampling_params)
        return result

def _sync_release_memory(engine,mode):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if mode == 'sleep':
            engine.release_memory_occupation(tags=['weights', 'kv_cache'])
            snapshot('after sleep inference engine')
            time.sleep(1)
            engine.resume_memory_occupation(tags=['weights', 'kv_cache'])
            snapshot('after wakeup inference engine')
    finally:
        loop.close()


async def _release_in_thread(engine,mode):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_release_memory, engine,mode)

async def run_server(server_args):
    inference = InferenceEngine(**dataclasses.asdict(server_args))
    snapshot('after load inference engine')
    await _release_in_thread(inference.engine,mode='sleep')
    snapshot('after sleep inference engine')
    # await _release_in_thread(inference.engine,mode='wakeup')
    # snapshot('after wakeup inference engine')
    #inference.
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 100
    from transformers import AutoTokenizer
    p = '/mnt/seed17/001688/zhaoping/LLMs/Qwen3-8B'
    tok = AutoTokenizer.from_pretrained(p)
    import pandas as pd
    path = "/mnt/seed17/001688/zhaoping/Data/AM-Thinking-v1-RL-Dataset/math_train.parquet"
    df = pd.read_parquet(path)
    infos = []
    for i,x in enumerate(df[['prompt','reward_model']].values[:]):
        prompt = x[0]
        prompt = tok.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
        #print(prompt)
        infos.append(prompt)
    prompts = infos[:64]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95,'max_new_tokens':4096*1}

    # Run the generation tasks concurrently in async mode.
    tasks = []
    for prompt in prompts:
        task = asyncio.create_task(inference.generate(prompt, sampling_params))
        tasks.append(task)
    
    outputs = []
    total_tokens = []
    # Get and print the result
    for task in tasks:
        await task
        while True:
            if not task.done():
                time.sleep(1)
            else:
                result = task.result()
                outputs.append(result)
                total_tokens.append(result['meta_info']['completion_tokens'])
                #print(f"Generated text: {result['text']}")
                break
            
    import json
    with open('/home/sglang/examples/runtime/engine/test_skyo1_32k_qwen3-8b-async-tp1.json','w') as f:
        json.dump(outputs,f,indent=4,ensure_ascii=False)
    print(f"mean tokens: {sum(total_tokens) / len(total_tokens)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.disable_cuda_graph = True
    args.attention_backend = 'fa3'
    args.tensor_parallel_size= 1
    args.enable_memory_saver = True
    args.device = 'musa'
    args.enable_nan_detection = False
    args.disable_overlap_scheduler = True
    print(args)
    server_args = ServerArgs.from_cli_args(args)
    asyncio.run(run_server(server_args))
