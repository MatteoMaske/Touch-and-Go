CUDA_VISIBLE_DEVICES=0,1 python train_CMC_touch.py --model resnet18t2 --batch_size 64 --num_workers 8 --data_folder dataset/ --model_path ckpt/cmc --dataset object_folder_balanced --learning_rate 0.05 --wandb --wandb_name matteomascherin-university-of-trento --nce_k 2048 --resume /home/matteomascherin/researchProject/Touch-and-Go/Visuo-tactile-contrastive-learning/ckpt/cmc/memory_nce_2048_resnet18t2_lr_0.05_decay_0.0001_bsz_64_dataset_ofb_view_Touch/ckpt_epoch=115.ckpt