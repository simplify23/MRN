common=dict(
    batch_max_length = 25,
    imgH = 32,
    imgW = 100,
    manual_seed=111,
    # character="../dataset/MLT2017/val_gt/mlt_2017_val",
)


""" Model Architecture """
model=dict(
    model_name="CRNN",
    Transformation = "None",
    FeatureExtraction = "ResNet",
    SequenceModeling = "BiLSTM",
    Prediction = "Attn",
    num_fiducial=20,
    input_channel=4,
    output_channel=512,
    hidden_size=256,
)


""" Optimizer """
optimizer=dict(
    schedule="super", #default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER
    optimizer="adam",
    lr=0.0005,
    sgd_momentum=0.9,
    sgd_weight_decay=0.000001,
    rho=0.95,
    eps=1e-8,
    lr_drop_rate=0.1
)


""" Data processing """
train = dict(
    exp_name="CRNN_baseline",  # Where to store logs and models
    saved_model="",  # "path to model to continue training"
    Aug="ABINet",  # |None|Blur|Crop|Rot|ABINet
    workers=4,
    train_data="../dataset/MLT2017/mlt_2017_train_Latin",
    valid_data="../dataset/MLT2017/val_gt/mlt_2017_val",
    select_data="/",
    batch_ratio="1.0",
    total_data_usage_ratio="1.0",
    NED=None,
    batch_size=384,
    num_iter=200000,
    val_interval=2000,
    log_multiple_test=None,
    FT="init",
    grad_clip=5,
    self_pre="RotNet",  # whether to use `RotNet` or `MoCo` pretrained model.
    semi="None", #|None|PL|MT|
    MT_C=1,
    MT_alpha=0.999,
    model_for_PseudoLabel="",
)


test=dict(
    eval_data="../dataset/MLT2017/",
    eval_type="IL_STR",
    workers=4,
    batch_size=512,
    saved_model="saved_models/CRNN_real/best_score.pth", #saved_models/CRNN_real/best_score.pth
    log_multiple_test=None,
    NED=True,
    Aug=None,#|None|Blur|Crop|Rot|
    semi="None",

)

