To launch a simple training:

```bash
python train_light_farl_plusv2.py --pretrained_model_name_or_path stablediffusionapi/realistic-vision-v51 --data_file data/dataset/dataset.csv --image_encoder_path Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64 --noise_offset 1e-6 --train_batch_size 1 --learning_rate 1e-15 --allow_tf32