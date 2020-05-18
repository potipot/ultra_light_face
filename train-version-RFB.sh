#!/usr/bin/env bash
model_root_path="./models/RFB-full_ds-Adam-1e-5"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python -u train.py \
  --datasets \
  ./data/ir_tufts_video_flir \
  --net \
  RFB \
  --validation_epochs 20 \
  --num_epochs \
  1000 \
  --lr \
  1e-5 \
  --resume \
  "models/pretrained/RFB-640-gray-resume.pth" \
  --batch_size \
  16 \
  --input_size \
  640 \
  --checkpoint_folder \
  ${model_root_path} \
  --num_workers \
  4 \
  --log_dir \
  ${log_dir} \
  --cuda_index \
  0 \
  --optimizer_type \
  Adam \
  2>&1 | tee "$log"
