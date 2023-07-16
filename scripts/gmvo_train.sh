# CHECKPOINT_DIR=checkpoints_flow/singapore-gmflow-scale2 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python main_flow.py \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage singapore_vo \
# --batch_size 1 \
# --lr 4e-4 \
# --image_size 128 128 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --with_speed_metric \
# --val_freq 10000 \
# --save_ckpt_freq 10000 \
# --num_steps 100000 \
# --image_dir "/tmp/Singapore" \
# --label_dir "/tmp/Singapore" \
# --debug \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
# --resume gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \

CHECKPOINT_DIR=checkpoints_flow/singapore-gmflow-scale2 && \
mkdir -p ${CHECKPOINT_DIR} && \
python main_flow.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage singapore_vo \
--val_dataset singapore_vo \
--batch_size 1 \
--lr 2e-4 \
--image_size 32 32 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 \
--with_speed_metric \
--val_freq 1 \
--save_ckpt_freq 10000 \
--num_steps 30000 \
--summary_freq 1 \
--image_dir "/tmp/Singapore" \
--label_dir "/tmp/Singapore" \
--debug \
--pred_bidir_flow \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log