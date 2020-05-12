#!/usr/bin/env bash
model_root_path="./models/train-version-RFB"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python -u train.py \
  --datasets \
  ./data/wider_face_add_lm_10_10 \
  --validation_dataset \
  ./data/wider_face_add_lm_10_10 \
  --net \
  RFB \
  --num_epochs \
  200 \
  `--milestones` \
  `95,150` \
  --lr \
  1e-4 \
  --resume \
  "models/pretrained/version-RFB-640.pth" \
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
