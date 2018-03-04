python train_vgg16_place365.py --trainset_dir ./data/indoor \
                        --batch_size 32 \
                        --n_epochs 25 \
                        --base_lr 0.001 \
                        --lr_decay_step 10000 \
                        --lr_decay_rate 0.5 \
                        --checkpoint_step 1000 \
                        --checkpoint_save_path ./checkpoint/vgg16_encoder \
                        --keep_checkpoint_num 80 \
                        --evalset_dir ./data/eval_indoor \
                        --eval_step 500 \
                        --eval_batch_size 32 \
                        --eval_save_path ./eval_results/eval_results_ \
                        --summary_step 50 \
                        --use_gpu True \
                        --gpu_fraction 1.0 \
                        --optimizer MOM \
                        --pretrained_model ./vgg16_pretrained/vgg16_place365.ckpt
# default summary_train_log_dir
# default summary_eval_log_dir
# default print step = 10
# default no pretrained model used
# default ADAM optimizer
