    # --config-file configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
    # --input /home/hrw/datasets/cityscapes/data/leftImg8bit/val/frankfurt/frankfurt_000000_003357_leftImg8bit.png \
    # --input /home/hrw/datasets/zerowaste-f/test/data/01_frame_000680.PNG \
    # MODEL.WEIGHTS maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth \
CUDA_VISIBLE_DEVICES=0 python demo/demo.py \
    --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
    --finetuning-checkpoint <<PATH TO CHECKPOINT>> \
    --confidence-threshold 0.5 \
    --input <<PATH TO IMG>> \
    --output <<OUT IMG PATH>> \
    --opts \
    MODEL.WEIGHTS maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES <<NUM CLASSES>> \
    MODEL.PIXEL_MEAN "[<<MEAN_CHANNEL1>>, <<MEAN_CHANNEL2>>, <<MEAN_CHANNEL3>>]" \
    MODEL.PIXEL_STD "[<<STD_CHANNEL1>>, <<STD_CHANNEL2>>, <<STD_CHANNEL3>>]" \
    SOLVER.IGNORE_FIX "['mask_embed', 'class_embed', 'bbox_embed']" \
    DATASETS.TEST "(<<DATASET_NAME_VAL>>,)" \
    INPUT.IMAGE_SIZE <<INPUT IMAGE SIZE>> \
    INPUT.MIN_SCALE 0.9 \
    INPUT.MAX_SCALE 2.0 \
    INPUT.DATASET_MAPPER_NAME <<DATASET_MAPPER NAME>> \
    USE_LORA True \
    LORA_DEFORMABLE_TARGETS "['sampling_offsets', 'value_proj']" \
    LORA_TARGETS "['q', 'v']" \
    LORA_RANK 8 \
    LORA_ALPHA 8