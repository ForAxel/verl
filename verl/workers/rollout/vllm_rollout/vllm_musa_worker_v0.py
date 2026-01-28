#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm_musa project.
# Adapted from vllm-project/vllm/vllm/worker/worker.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
from typing import Dict, List, Optional, Tuple, Type

import os
import time
from datetime import datetime
from distutils.version import StrictVersion

import torch
import torch.distributed
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import (GiB_bytes, MemorySnapshot, memory_profiling)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner

from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.worker_base import WorkerBase
from vllm.worker.worker import Worker
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)

logger = init_logger(__name__)

class MTGPUWorker(Worker):
    """A worker class that executes (a partition of) the model on a MTGPU.
    Each worker is associated with a single MTGPU. The worker is responsible for
    maintaining the KV cache and executing the model on the MTGPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[ModelRunnerBase]] = None,
    ) -> None:
        # Register ops and patch when worker init.
        from vllm_musa import ops  # noqa: F401
        from vllm_musa import patch  # noqa: F401

        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}
        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            self.torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                self.torch_profiler_trace_dir,
            )

            def trace_handler(prof):
                begin = time.monotonic()
                current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
                rank = torch.distributed.get_rank()
                save_dir = os.path.join(self.torch_profiler_trace_dir, current_time)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                prof.export_chrome_trace(f"{save_dir}/profile_rank{rank}.json")
                print(
                    f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
                )

            if StrictVersion(torch.__version__) < StrictVersion('2.5.0'):
                activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MUSA,
                    ]
            else:
                activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.PrivateUse1,
                    ]

            self.profiler = torch.profiler.profile(
                activities=activities,
                profile_memory=True, 
                record_shapes=True,
                with_flops=True,
                with_modules=True,
                with_stack=True, 
                on_trace_ready=trace_handler,
            )
        else:
            self.profiler = None

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
            
        for cache in self.gpu_cache[0]:
            cache.zero_()    

    def init_device(self) -> None:
        if self.device_config.device.type == "musa":
            # os.environ["MCCL_PROTOS"] = "2"
            self.device = torch.device(f"musa:{self.rank}") 
            torch.musa.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.musa.empty_cache()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        self.baseline_snapshot = MemorySnapshot()
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    
    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.musa.empty_cache()
        torch.musa.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.musa.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.baseline_snapshot,
                weights_memory=self.model_runner.model_memory_usage) as result:
            self.model_runner.profile_run()

        self._assert_memory_footprint_increased_during_profiling()

        memory_for_current_instance = total_gpu_memory * \
            self.cache_config.gpu_memory_utilization
        available_kv_cache_memory = (memory_for_current_instance -
                                     result.non_kv_cache_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        msg = (f"Memory profiling takes {result.profile_time:.2f} seconds\n"
               "the current vLLM instance can use "
               "total_gpu_memory "
               f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
               " x gpu_memory_utilization "
               f"({self.cache_config.gpu_memory_utilization:.2f})"
               f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
               "model weights take "
               f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
               " non_torch_memory takes "
               f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
               " PyTorch activation peak memory takes "
               f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
               " the rest of the memory reserved for KV Cache is "
               f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB.")

        logger.info(msg)
        # Final cleanup
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks
    
    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()
        # print(
        #     self.profiler.key_averages().table(sort_by="self_musa_time_total"))

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        free_gpu_memory, total = torch.musa.mem_get_info()
        cuda_memory = total - free_gpu_memory
        assert self.baseline_snapshot.cuda_memory < cuda_memory, (
            "Error in memory profiling. "
            f"Initial used memory {self.baseline_snapshot.cuda_memory}, "
            f"currently used memory {cuda_memory}. "
            f"This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        return True

def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend="mccl")
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)

