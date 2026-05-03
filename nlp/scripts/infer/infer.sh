script='/raid_igvozdarev/repo/nlp/scripts/infer/infer.py'
configs_dir='configs'
model_config='configs/config.yaml'          
chat_config='configs/chat_configs/config.yaml'
model_name='/raid_igvozdarev/raid/hf_models/Qwen3.5-27B'     # 'Qwen/Qwen3.5-27B'


export HF_TOKEN='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'
export HF_HUB_CACHE='/raid_igvozdarev/raid/hf_models'
export CUDA_VISIBLE_DEVICES='2,3'
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET

export NCCL_P2P_DISABLE=1
# export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

cd '/raid_igvozdarev/rail/notebooks/bench'


python $script \
    --engine vllm \
    --model_name $model_name \
    --model_config $model_config \
    --chat_config $chat_config \
    --conv_dataset fix_sample.jsonl \
    --batch_size 9999
