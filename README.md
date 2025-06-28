To launch a simple training:

```bash
python train_light_farl_plusv2.py --pretrained_model_name_or_path stablediffusionapi/realistic-vision-v51 --data_file data/dataset/dataset.csv --image_encoder_path Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64 --noise_offset 0 --allow_tf32 --use_t2i --validation_prompt "A young boy riding on the back of a brown and white horse next to a red fire hydrant." --validation_image 0.jpg --validation_steps 25