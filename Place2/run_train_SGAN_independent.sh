python train_SGAN_independent.py --trainset_dir ./data/indoor \
                        --batch_size 16 \
                        --n_epochs 50 \
                        --optimizer ADAM \
                        --G_base_lr 0.0002 \
                        --D_base_lr 0.0002 \
                        --lr_decay_step 900000 \
                        --lr_decay_rate 1.0 \
                        --advloss_weight 1.0 \
                        --condloss_weight 1.0 \
                        --entloss_weight 1.0 \
                        --z0_dim 1000 \
                        --z1_dim 500 \
                        --z2_dim 100 \
                        --E_pretrained ./vgg16_pretrained/encoder-32000.ckpt \
                        --checkpoint_step 1000 \
                        --checkpoint_save_path ./checkpoint/new_SGAN_independent \
                        --keep_checkpoint_num 3 \
                        --real_img_dir ./results/real_img_new/ \
                        --gen_img_dir ./results/gen_img_new/ \
                        --merge_nh 4 \
                        --merge_nw 4 \
                        --save_img_step 500\
                        --summary_step 100 \
                        --summary_train_log_dir ./train_SGAN_new_log \
                        --use_gpu True \
                        --gpu_fraction 1.0

#--G0_pretrained checkpoint/new_SGAN_independent20170110-144833.ckpt-26988 \
#--G1_pretrained checkpoint/new_SGAN_independent20170110-144833.ckpt-26988 \
#--G2_pretrained checkpoint/new_SGAN_independent20170110-144833.ckpt-26988 \

# default print step = 10
# default no pretrained model used
# --G1_pretrained ./SGAN_pretrained/SGAN_G1.ckpt \
