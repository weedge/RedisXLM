[![licence](https://img.shields.io/github/license/weedge/redisxlm.svg)](https://github.com/weedge/redisxlm/blob/main/LICENSE)

> [!NOTE]
> redis embedded language model, available for stand-alone version only

## RedisXLM
- use rust/c/c++ impl redisxlm modules  

- redis x language model (load pre-trained model, instruction-tuned model); size (tiny|t, small|s, medium|m, large|l) with quantization;

- Load model types：
   1. embedding model
   2. generation(inference) model

- Third-party open source libraries used:
  1. https://github.com/karpathy/llama2.c (simple, inference Llama 2 in one file of pure C) 
  2. https://github.com/ggerganov/llama.cpp (Integrated nearly all open-source LLMs, including the following open-source LLMs)
  3. https://github.com/google/gemma.cpp (Google's open-source LLM)
  4. https://github.com/li-plus/chatglm.cpp (LLM open-sourced by the Tsinghua University community)
  5. https://github.com/QwenLM/qwen.cpp (Similar to chatglm, LLM open-sourced by Alibaba)

## Cases

## Reference
