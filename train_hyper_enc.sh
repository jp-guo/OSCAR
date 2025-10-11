bpp=""
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port main_train_hyper_enc.py \
    --pretrained_model_name_or_path="model_zoo/stable-diffusion-2-1" \
    --learning_rate=5e-5 \
    --gradient_accumulation_steps=1 \
    --max_train_steps 10000 \
    --checkpointing_steps 1000 \
    --mixed_precision='no' \
    --seed 123 \
    --tracker_project_name "hyper_enc_${bpp}" \
    --bpp "$bpp" \
    --hyper_dim 320