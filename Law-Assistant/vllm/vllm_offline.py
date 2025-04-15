# 实现vLLM的离线推理

from vllm_wrapper import vLLMWrapper

model = "qwen/Qwen-1_8B-Chat-Int8"

vllm_model = vLLMWrapper(model,
                        quantization = 'gptq',  # 千问使用gptq这个库去做量化的
                        dtype="float16",
                        tensor_parallel_size=1, # GPU的个数，支持多卡并行推理
                        gpu_memory_utilization=0.6)  # 调整vllm的显存占用比例

history=None 
while True:
    Q=input('提问:')
    response, history = vllm_model.chat(query=Q,
                                        history=history)
    print(response)
    history=history[:20]