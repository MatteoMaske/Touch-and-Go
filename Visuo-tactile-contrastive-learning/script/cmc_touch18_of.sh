CUDA_VISIBLE_DEVICES=0,1 python train_CMC_touch.py --model resnet18t2 --batch_size 64 --num_workers 8 --data_folder dataset/ --model_path ckpt/cmc --dataset object_folder_balanced --learning_rate 0.05 --wandb --wandb_name matteomascherin-university-of-trento --supconloss