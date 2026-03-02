export NCCL_DEBUG=WARN
PORT=29501

DATAPATH=./data_example/colmap_processed/colmap_data_path.json
EXP_NAME=colmap_eval
CKPT_PATH=./checkpoints/dl3dv_full.pt
NUM_VIEWS=16 # 16, 64
CONFIG=./configs/dl3dv_full.yaml

torchrun --nproc_per_node 1 --nnodes 1 \
        --rdzv_id 18638 --rdzv_backend c10d \
        --rdzv_endpoint localhost:${PORT}  \
        inference.py ${CONFIG} \
        -s evaluation true -s evaluation_out_dir ./evaluation/${EXP_NAME}/$(basename "$DATAPATH")_${NUM_VIEWS} \
        -s eval_dataset_path $DATAPATH \
        -s training.target_has_input False \
        -s training.wandb_exp_name ${EXP_NAME} \
        -s training.batch_size_per_gpu 1 -s training.dataset_path $DATAPATH \
        -s training.num_views 35 -s training.num_input_views ${NUM_VIEWS} -s training.num_target_views ${NUM_VIEWS} -s training.num_virtual_views ${NUM_VIEWS} \
        -s training.checkpoint_dir ./checkpoints/${EXP_NAME} -s model.use_anything False -s model.act_ckpt False -s kmeans_input True -s training.reset_training_state False -s metrics_only False -s training.data_repeat 1 -s training.perceptual_loss_weight 0.0 -s num_frames 8 -s training.view_selector.type kmeans \
        -s sp_size 1 -s model.gaussians.usage_threshold 0.001 -s training.torch_compile True --load ${CKPT_PATH}
