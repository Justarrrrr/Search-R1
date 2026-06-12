#!/usr/bin/env bash
# Search-R1 Level 2 多模态 GRPO 训练脚本（最小可启动版本，不追求收敛）。
#
# 前置条件：
#   1) 已构建 COCO 图像索引：bash scripts/build_image_index_coco.sh
#   2) 已生成训练 parquet：python scripts/data_process/coco_image_qa.py ...
#   3) 已启动图像检索服务：bash retrieval_launch_image.sh （监听 8000 端口）
#   4) 已注册 reward 函数：verl/utils/reward_score/coco_image_qa.py + main_ppo.py 路由
#
# 与原 train_grpo.sh 的关键差异：
#   - BASE_MODEL 换成 Qwen3-VL-4B-Instruct（多模态）
#   - data_source 名设为 coco_image_qa，由 _select_rm_score_fn 路由到 coco_image_qa.compute_score_em
#   - retriever url 仍走 8000 端口，但服务端是 image_retrieval_server
#   - 训练规模缩小（batch / steps）以适配 3090 + 4B 模型
#
# 注意：vllm 当前版本对 Qwen3-VL 的支持仍在演进，若 rollout 报错，
# 可改用 actor_rollout_ref.rollout.name=hf 走 HF generate 路径（更慢但兼容性好）。

set -e

# GPU1：避开 GPU0 上的检索服务（占用约 1GB 显存）
export CUDA_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
# 防止 NCCL 单卡场景下 hang
export NCCL_TIMEOUT=300
export NCCL_DEBUG=WARN

WAND_PROJECT='Search-R1-MM'
# 正式训练：Qwen3-0.6B + 图像检索，跑完整 1 epoch 产出 reward 曲线
export BASE_MODEL='/var/lib/container/dataset/yxqiu/models/Qwen3-0.6B'
export EXPERIMENT_NAME=coco-search-r1-grpo-qwen3-0.6b-full-v1

DATA_DIR='data/coco_image_qa'
TRAIN_FILE=$DATA_DIR/train.parquet
TEST_FILE=$DATA_DIR/test.parquet

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] $TRAIN_FILE not found."
    echo "Run: python scripts/data_process/coco_image_qa.py \\"
    echo "       --annotation_path /var/lib/container/dataset/yxqiu/datasets/coco/annotations/captions_val2017.json \\"
    echo "       --out_dir $DATA_DIR"
    exit 1
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:18000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
