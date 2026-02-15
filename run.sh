conda activate turbodiffusion
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=turbodiffusion
export HF_HOME=~/.cache/huggingface

python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-1.3B \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --resolution 480p \
    --prompt "A giant panda hanging with an axe trying to break the tree wall in front of him." \
    --num_samples 1 \
    --num_steps 4 \
    --quant_linear \
    --attention_type sla \
    --sla_topk 0.1

#   ┌────────────────────┬──────┬────────────────────────────────────────┐
#   │        技术        │ 状态 │                  说明                  │
#   ├────────────────────┼──────┼────────────────────────────────────────┤
#   │ rCM                │ ✅   │ 4 步采样（核心加速）                   │
#   ├────────────────────┼──────┼────────────────────────────────────────┤
#   │ SLA                │ ✅   │ 稀疏线性注意力                         │
#   ├────────────────────┼──────┼────────────────────────────────────────┤
#   │ INT8 量化          │ ✅   │ --quant_linear，量化 checkpoint        │
#   ├────────────────────┼──────┼────────────────────────────────────────┤
#   │ 自定义 Triton Norm │ ✅   │ 修了 kernel bug 后生效                 │
#   ├────────────────────┼──────┼────────────────────────────────────────┤
#   │ SageSLA            │ ❌   │ SpargeAttn 未装，用的 sla 不是 sagesla │
#   └────────────────────┴──────┴────────────────────────────────────────┘