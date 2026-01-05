import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['musa_LAUNCH_BLOCKING'] = '1'

import torch
import torch.distributed as dist

# patch for contiguous tensor
def patch_tensor_operations():
    original_chunk = torch.chunk
    original_split = torch.split
    original_transpose = torch.transpose
    
    def safe_chunk(tensor, chunks, dim=0):
        result = original_chunk(tensor, chunks, dim)
        return [t.contiguous() if not t.is_contiguous() else t for t in result]
    
    def safe_split(tensor, split_size, dim=0):
        result = original_split(tensor, split_size, dim)
        return [t.contiguous() if not t.is_contiguous() else t for t in result]
    
    def safe_transpose(input, dim0, dim1):
        result = original_transpose(input, dim0, dim1)
        return result.contiguous() if not result.is_contiguous() else result
    
    torch.chunk = safe_chunk
    torch.split = safe_split
    torch.transpose = safe_transpose
    print(f"====== Add tensor contiguous patch ======")

# apply patch
patch_tensor_operations()

# MUSA check
def pre_distributed_check():
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU Num: {torch.musa.device_count()}")
    for i in range(torch.musa.device_count()):
        print(f"GPU {i}: {torch.musa.get_device_name(i)}")
    if hasattr(torch, 'musa'):
        print("MUSA support enable")
    else:
        print("Do not have MUSA support")

pre_distributed_check()