docker run --rm -it   --name qwen27vllm_2   --gpus all   --shm-size=32g   -p 7866:7866   \
    -v /raid/hf_models/Qwen3.5-27B:/model \
    --entrypoint /usr/local/bin/vllm   -e NCCL_P2P_DISABLE=1   -e NCCL_IB_DISABLE=1 \
    -e VLLM_DISABLE_COMPILE_CACHE=1   -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    vllm/vllm-openai:minimax27-cu130   serve /model   \
    --host 0.0.0.0   --port 7866   --served-model-name qwen35_27b  \
    --tensor-parallel-size 2   --disable-custom-all-reduce   --max-model-len 24000  \
    --gpu-memory-utilization 0.85   --dtype auto   --trust-remote-code  \
    --enable-auto-tool-choice   --tool-call-parser hermes   --enable-lora \
    --max-lora-rank 64   --lora-modules gui_lora=/lora   --api-key EMPTY