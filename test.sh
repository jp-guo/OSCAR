export CUDA_VISIBLE_DEVICES=0

dataset=""

accelerate launch --main_process_port main_test.py \
    -i "dataset/test/$dataset" \
    -o "results/${dataset}" \
    --pretrained_model_name_or_path="model_zoo/stable-diffusion-2-1" \
    --seed 123 \
    --oscar_path "model_zoo/oscar.pkl"