export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh \
  --rounds 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --victim-model-id internlm/internlm2-chat-7b \
  --attempts 20 \
  --gpu-memory-utilization 0.40 \
  --shared-gpu-memory-utilization 0.55 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --dataset-size 1000 \
  --output-dir results/benchmarks/Internlm2_subset-8_$(date +%F_%H-%M-%S)_4g \
  --start-idx 0 