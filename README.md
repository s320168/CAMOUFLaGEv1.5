To launch a simple training:

```bash
python train_light_farl_plusv2.py --pretrained_model_name_or_path stablediffusionapi/realistic-vision-v51 --data_file data/dataset/dataset.csv --image_encoder_path Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64 --noise_offset 0 --allow_tf32 --use_t2i --validation_prompt "A young boy riding on the back of a brown and white horse next to a red fire hydrant." --validation_image 0.jpg --checkpointing_steps 50000
```

```bash
python train_light_farl_plusv2.py --pretrained_model_name_or_path stablediffusionapi/realistic-vision-v51 --data_file data/dataset/dataset.csv --image_encoder_path Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64 --noise_offset 0 --allow_tf32 --use_t2i --validation_prompt "A woman and a young girl posing for a picture together with other people in the background at a park." --validation_image 1.jpg --checkpointing_steps 50000
```

```bash
python train_light_farl_plusv2.py --pretrained_model_name_or_path stablediffusionapi/realistic-vision-v51 --data_file database.csv --image_encoder_path Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64 --noise_offset 0 --allow_tf32 --use_t2i --validation_prompt "A woman wearing a headset in front of a blue screen with a bright light in the middle of it." --validation_image data/2.jpg  --checkpointing_steps 50000
```