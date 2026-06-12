# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.utils.import_utils import is_vllm_available, is_megatron_core_available

from .base import BaseShardingManager
from .fsdp_ulysses import FSDPUlyssesShardingManager

AllGatherPPModel = None
MegatronVLLMShardingManager = None
FSDPVLLMShardingManager = None

# 注：原实现仅以 is_vllm_available() 决定是否 import，但 verl 自带的
# verl/third_party/vllm 只支持 vllm 0.3.1/0.4.2/0.5.4/0.6.3，遇到更新版本（如 0.10.x）
# 会在 import 阶段抛 ValueError。这里做兼容处理：
# 当前只跑 hf rollout 路径，FSDP/Megatron VLLM 适配层加载失败不应阻塞主流程。
if is_megatron_core_available() and is_vllm_available():
    try:
        from .megatron_vllm import AllGatherPPModel, MegatronVLLMShardingManager
    except (ImportError, ValueError) as _e:
        import warnings
        warnings.warn(f"[sharding_manager] skip megatron_vllm import: {_e}")

if is_vllm_available():
    try:
        from .fsdp_vllm import FSDPVLLMShardingManager
    except (ImportError, ValueError) as _e:
        import warnings
        warnings.warn(f"[sharding_manager] skip fsdp_vllm import: {_e}")
