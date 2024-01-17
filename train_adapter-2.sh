CUDA_VISIBLE_DEVICES=1 python train_net.py \
    --num-gpus 1 \
    --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
    MODEL.WEIGHTS maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES 12 \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.001 \
    SOLVER.IGNORE_FIX "['mask_embed', 'class_embed', 'bbox_embed', 'pixel_decoder_self_attention_adapter', 'decoder_cross_attn_adapter', 'decoder_self_attn_adapter']" \
    SOLVER.MAX_ITER 11090 \
    SOLVER.CHECKPOINT_PERIOD 1109 \
    SOLVER.STEPS "(5545,)" \
    TEST.EVAL_PERIOD 1109 \
    DATASETS.TRAIN "('xray-waste-train',)" \
    DATASETS.TEST "('xray-waste-val',)" \
    INPUT.IMAGE_SIZE 450 \
    INPUT.MIN_SCALE 0.9 \
    INPUT.MAX_SCALE 2.0 \
    INPUT.DATASET_MAPPER_NAME 'xray-waste' \
    OUTPUT_DIR output/xray-waste_adapter-2_10-epochs \
    USE_ADAPTERS True