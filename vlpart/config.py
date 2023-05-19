# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_vlpart_config(cfg):
    """
    Add config for VLPart.
    """
    # dino
    cfg.DINO = CN()
    cfg.DINO.MODEL_TYPE = 'dino_vits8'
    cfg.DINO.STRIDE = 4
    cfg.DINO.LAYER = 11
    cfg.DINO.THRESH = 0.05
    cfg.DINO.FACET = 'key'
    cfg.DINO.BUILD_BASEDATA = False
    cfg.DINO.BASEDATA_SAVE_DIR = 'output_basedata'
    cfg.DINO.BASEDATA_ANN_PATH = 'datasets/pascal_part/train_base_one.json'
    cfg.DINO.BASEDATA_IMS_PATH = 'datasets/pascal_part/VOCdevkit/VOC2010/JPEGImages/'
    cfg.DINO.BASE_OBJ_CAT = [-1]
    cfg.DINO.PIXEL_NORM = True
    cfg.DINO.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.DINO.PIXEL_STD = [0.229, 0.224, 0.225]
    cfg.DINO.MIN_SIZE_TEST = 224

    # ann generator
    cfg.MODEL.ANN_GENERATOR = False
    cfg.OUTPUT_ANN_DIR = 'datasets/grouping.json'

    # parsed part
    cfg.MODEL.ROI_BOX_HEAD.PART_LOSS_TYPE = 'max_score'
    cfg.MODEL.ROI_BOX_HEAD.PARSED_PART_LOSS_WEIGHT = 0.1

    # joint training
    cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS_GROUP = [False]
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS_GROUP = [True]
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP = ['datasets/metadata/lvis_v1_train_cat_info.json']
    cfg.MODEL.ROI_HEADS.MASK_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.MULT_OBJECT_SCORE = False

    # evaluation
    cfg.MODEL.EVAL_ATTR = False
    cfg.MODEL.EVAL_PER = False
    cfg.MODEL.EVAL_PROPOSAL = False
    cfg.MODEL.LOAD_PROPOSALS_TEST = False
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False  # For demo only

    # Open-vocabulary classifier
    cfg.WITH_IMAGE_LABELS = False  # Turn on co-training with classification data
    cfg.MODEL.ROI_BOX_HEAD.NOUNS_LOSS_WEIGHT = 0.01
    cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = False  # Use fixed classifier for open-vocabulary detection
    cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS_GROUP = False
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP = ['datasets/metadata/lvis_v1_clip_a+cname.npy']
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 1024
    cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP = 50.0
    cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0  # >= 0: not use
    cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False  # CenterNet2
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False  # Federated Loss
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = 'datasets/metadata/lvis_v1_train_cat_info.json'
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    # Classification data configs
    cfg.MODEL.ROI_BOX_CASCADE_HEAD.IMAGE_LABEL_LOSSES = ['max_size', 'max_size', 'max_size']
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = 'max_size'  # max, softmax, sum
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT = 0.1
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE = 1.0
    cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX = False  # Used for image-box loss and caption loss
    cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 128  # num proposals for image-labeled data
    cfg.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP = False  # Used for WSDDN
    cfg.MODEL.ROI_BOX_HEAD.WITH_REFINEMENT_SCORE = False  # Used for OICR
    cfg.MODEL.ROI_BOX_HEAD.REFINEMENT_IOU = 0.5  # Used for OICR
    cfg.MODEL.ROI_BOX_HEAD.REGION_LOSS_WEIGHT = 3.0  # Used for OICR
    cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP = False  # Used for WSDDN
    cfg.MODEL.ROI_BOX_HEAD.SOFTMAX_WEAK_LOSS = False  # Used when USE_SIGMOID_CE is False

    # Caption data configs
    cfg.MODEL.TEXT_ENCODER_TYPE = 'ViT-B/32'
    cfg.MODEL.TEXT_ENCODER_DIM = 512
    cfg.MODEL.WITH_CAPTION = False
    cfg.MODEL.CAP_BATCH_RATIO = 4  # Ratio between detection data and caption data
    cfg.MODEL.SYNC_CAPTION_BATCH = False  # synchronize across GPUs to enlarge # "classes"
    cfg.MODEL.ROI_BOX_HEAD.WITH_CAPTION_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.CAPTION_LOSS_WEIGHT = 1.0  # Caption loss weight
    cfg.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT = 0.125  # Caption loss hyper-parameter

    # dynamic class sampling when training with 21K classes
    cfg.MODEL.DYNAMIC_CLASSIFIER = False
    cfg.MODEL.NUM_SAMPLE_CATS = 50

    # Different classifiers in testing, used in cross-dataset evaluation
    cfg.MODEL.RESET_CLS_TESTS = False
    cfg.MODEL.TEST_CLASSIFIERS = []
    cfg.MODEL.TEST_NUM_CLASSES = []
    cfg.MODEL.DATASET_LOSS_WEIGHT = []

    # Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'T'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (1, 2, 3)  # FPN stride 8 - 32

    cfg.MODEL.TIMM = CN()
    cfg.MODEL.TIMM.BASE_NAME = 'resnet50'
    cfg.MODEL.TIMM.OUT_LEVELS = (3, 4, 5)
    cfg.MODEL.TIMM.NORM = 'FrozenBN'
    cfg.MODEL.TIMM.FREEZE_AT = 0
    cfg.MODEL.TIMM.PRETRAINED = False

    # Multi-dataset dataloader
    cfg.DATALOADER.DATASET_RATIO = [1, 1]  # sample ratio
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = False  # Always true when multi-dataset is enabled
    cfg.DATALOADER.DATASET_ANN = ['box', 'box']  # Annotation type of each dataset
    cfg.DATALOADER.STRONG_AUG_ON_PARSED = False
    cfg.DATALOADER.USE_DIFF_BS_SIZE = False  # Use different batchsize for each dataset
    cfg.DATALOADER.DATASET_BS = [8, 32]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_INPUT_SIZE = [896, 384]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_INPUT_SCALE = [(0.1, 2.0), (0.5, 1.5)]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_MIN_SIZES = [(640, 800), (320, 400)]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_MAX_SIZES = [1333, 667]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.USE_TAR_DATASET = False  # for ImageNet-21K, directly reading from unziped files
    cfg.DATALOADER.TARFILE_PATH = 'datasets/imagenet/metadata-22k/tar_files.npy'
    cfg.DATALOADER.TAR_INDEX_DIR = 'datasets/imagenet/metadata-22k/tarindex_npy'
    cfg.DATALOADER.CAPTION_PARSER = False

    # Custom solver
    cfg.SOLVER.USE_CUSTOM_SOLVER = False
    cfg.SOLVER.OPTIMIZER = 'SGD'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0  # Used in DETR
    cfg.SOLVER.CUSTOM_MULTIPLIER = 1.0  # Used in DETR
    cfg.SOLVER.CUSTOM_MULTIPLIER_NAME = []  # Used in DETR

    # EfficientDetResizeCrop config
    cfg.INPUT.CUSTOM_AUG = ''
    cfg.INPUT.TRAIN_SIZE = 640
    cfg.INPUT.TEST_SIZE = 640
    cfg.INPUT.SCALE_RANGE = (0.1, 2.)

    # FP16
    cfg.FP16 = False
    cfg.FIND_UNUSED_PARAM = True

    # visualize
    cfg.VIS = CN()
    cfg.VIS.SCORE = True
    cfg.VIS.BOX = True
    cfg.VIS.MASK = True
    cfg.VIS.LABELS = True
