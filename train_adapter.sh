CUDA_VISIBLE_DEVICES=1 python train_net.py \
    --num-gpus 1 \
    --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
    MODEL.WEIGHTS maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES <<NUM CLASSES>> \
    MODEL.PIXEL_MEAN "[<<MEAN_CHANNEL1>>, <<MEAN_CHANNEL2>>, <<MEAN_CHANNEL3>>]" \
    MODEL.PIXEL_STD "[<<STD_CHANNEL1>>, <<STD_CHANNEL2>>, <<STD_CHANNEL3>>]" \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.001 \
    SOLVER.IGNORE_FIX "['mask_embed', 'class_embed', 'bbox_embed', 'pixel_decoder_self_attention_adapter', 'decoder_cross_attn_adapter', 'decoder_self_attn_adapter']" \
    SOLVER.MAX_ITER <<NUM OPTIMIZER STEPS>> \
    SOLVER.CHECKPOINT_PERIOD <<NUM STEPS FOR CHECKPOINT>> \
    SOLVER.STEPS "(<<LR_STEP1>>, <<LR_STEP2>>, ...)" \
    TEST.EVAL_PERIOD <<NUM STEPS FOR EVAL>> \
    DATASETS.TRAIN "(<<DATASET_NAME_TRAIN>>,)" \
    DATASETS.TEST "(<<DATASET_NAME_VAL>>,)" \
    INPUT.IMAGE_SIZE <<INPUT IMAGE SIZE>> \
    INPUT.MIN_SCALE 0.9 \
    INPUT.MAX_SCALE 2.0 \
    INPUT.DATASET_MAPPER_NAME <<DATASET_MAPPER NAME>> \
    OUTPUT_DIR <<OUTPUT FILE DIR>> \
    USE_ADAPTERS <<True or False>> \
    ADAPTER_NUM <<NUM ADAPTERS>> \
    ADAPTER_REDUCTION <<ADAPTER REDUCTION>>