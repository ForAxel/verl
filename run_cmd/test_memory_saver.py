# import torch

# p = '/mnt/seed-program-nas/001688/kechun.wu/tmp0119/tmp_logs/model_ori.pt'
# ori = torch.load(p,weights_only=False)

# p = '/mnt/seed-program-nas/001688/kechun.wu/tmp0119/tmp_logs/model_resume.pt'
# resume = torch.load(p,weights_only=False)

# for name in ori.keys():
#     tensor1,data_ptr1,shape1= ori[name]
#     tensor2,data_ptr2,shape2 = resume[name]
#     if torch.isnan(tensor1).sum():
#         print(f"ori {name} has nan:{torch.isnan(tensor1).sum():}")
#     if torch.isnan(tensor2).sum():
#         print(f"resume {name} has nan:{torch.isnan(tensor2).sum():}")
#     if not torch.equal(tensor1, tensor2):
#         print(f'{name} tensor is different, size={shape1}, diff avg={(tensor1-tensor2).mean()}')
#     if data_ptr1 != data_ptr2:
#         print(f'{name} data_ptr is different')

        
#for name, param in model.named_parameters():

#from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from transformers import AutoModel
import time
from torch_memory_saver import torch_memory_saver
torch_memory_saver.hook_mode = 'torch'
import torch

GPU_MEMORY_TYPE_WEIGHTS = 'weight'
ori = {}
resume = {}
with torch_memory_saver.region(GPU_MEMORY_TYPE_WEIGHTS):
    model = AutoModel.from_pretrained('/mnt/seed17/001688/zhaoping/LLMs/Qwen3-8B',dtype=torch.bfloat16).to('musa')
    for name, param in model.named_parameters():
        ori[name] = [param,param.data_ptr(),param.stride(),param.storage_offset(),param.size()]
    p1 = '/tmp/model1.pt'
    torch.save(ori,p1)
    
torch_memory_saver.pause(tag=GPU_MEMORY_TYPE_WEIGHTS)
time.sleep(2)
torch_memory_saver.resume(tag=GPU_MEMORY_TYPE_WEIGHTS)

for name, param in model.named_parameters():
    resume[name] = [param,param.data_ptr(),param.stride(),param.storage_offset(),param.size()]
p2 = '/tmp/model2.pt'
torch.save(resume,p2)


ori = torch.load(p1,weights_only=False)
resume = torch.load(p2,weights_only=False)

for name in ori.keys():
    tensor1,data_ptr1,stride1,offset1,shape1= ori[name]
    tensor2,data_ptr2,stride2,offset2,shape2 = resume[name]
    if torch.isnan(tensor1).sum():
        print(f"ori {name} has nan:{torch.isnan(tensor1).sum():}")
    if torch.isnan(tensor2).sum():
        print(f"resume {name} has nan:{torch.isnan(tensor2).sum():}")
    if not torch.equal(tensor1, tensor2):
        print(f'{name} tensor is different, size={shape1}, diff avg={(tensor1-tensor2).mean()}')
    if offset1 != offset2:
        print(f'{name} offset is different')
    if data_ptr1 != data_ptr2:
        print(f'{name} data_ptr is different')
    if stride1 != stride2:
        print(f'{name} stride is different')
