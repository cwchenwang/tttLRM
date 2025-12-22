export NCCL_DEBUG=WARN
PORT=29501

DATAPATH=/home/chenwang/workspace/projects/tttLRM-all/test_data/dl3dv10k_benchmark.txt
EXP_NAME=gs_dl3dvttt_lineardepth_full_fdist128to512_res536x960_v64_t48_mixed_16_32_64_5e-5_bs64
NUM_VIEWS=32 # 16, 64
OPTIM_ROUNDS=0

torchrun --nproc_per_node 4 --nnodes 1 \
        --rdzv_id 18638 --rdzv_backend c10d \
        --rdzv_endpoint localhost:${PORT}  \
        trainer_sp.py /home/chenwang/workspace/projects/tttLRM-all/tttLRM-public/configs/dl3dv.yaml \
        -s evaluation true -s evaluation_out_dir ./evaluation/${EXP_NAME}/$(basename "$DATAPATH")_${NUM_VIEWS}_thres0.005_optim_${OPTIM_ROUNDS} \
        -s eval_dataset_path $DATAPATH \
        -s training.target_has_input False \
        -s training.wandb_exp_name ${EXP_NAME} \
        -s training.batch_size_per_gpu 1 -s training.dataset_path $DATAPATH \
        -s training.num_views 35 -s training.num_input_views ${NUM_VIEWS} -s training.num_target_views ${NUM_VIEWS} -s training.num_virtual_views ${NUM_VIEWS} \
        -s training.checkpoint_dir ./checkpoints/${EXP_NAME} -s model.use_anything False -s model.act_ckpt False -s kmeans_input True -s training.reset_training_state False -s metrics_only False -s training.data_repeat 1 -s training.perceptual_loss_weight 0.0 -s num_frames 8 -s training.view_selector.type kmeans \
        -s sp_size 1 -s model.gaussians.usage_threshold 0.01 -s training.torch_compile True --load /home/chenwang/workspace/projects/tttLRM-all/dl3dvckpt_0000000000005500.pt -s model.gaussians.optimize_gs_rounds ${OPTIM_ROUNDS}