common=dict(
    exp_name="CRNN_der_syn_exp1",  # Where to store logs and models
    il="der",  # joint_mix ｜ joint_loader | base | lwf | wa | ewc ｜ der  | ems
    memory='random', # None | rehearsal | random | bag | score | loss_max | test
    memory_num=2000,
    batch_max_length = 25,
    imgH = 32,
    imgW = 256,
    manual_seed=111,
    start_task = 0
)


""" Model Architecture """
model=dict(
    model_name="CRNN",
    Transformation = "None",
    FeatureExtraction = "VGG",
    SequenceModeling = 'BiLSTM',
    Prediction = "CTC",
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
    milestones=[2000,4000],
    lrate_decay=0.1,
    rho=0.95,
    eps=1e-8,
    lr_drop_rate=0.1
)


""" Data processing """
train = dict(
    saved_model="",  # "path to model to continue training"
    Aug="None",  # |None|Blur|Crop|Rot|ABINet
    workers=4,
    ch_list = None,
    # lan_list = ["ArT","RCTW","ReCTS","LSVT","CTW"],
    lan_list=["Chinese", "Latin", "Japanese", "Korean", "Arabic", "Bangla"],
    # lan_list=["Chinese", "Japanese", "Korean","Latin", "Arabic", "Bangla",],
    # lan_list=["Arabic", "Chinese", "Latin","Japanese", "Bangla","Korean",],
    # train_data = "../../../test/ztl/IL/MLT17_IL/train_2017",
    train_data = '/share/test/ztl/IL/IL_CHSTR/train',
    # valid_data="../../../test/ztl/IL/MLT19_IL/test_2019",
    valid_datas=[
                 "../../../test/ztl/IL/MLT17_IL/test_2017",
                 "../../../test/ztl/IL/MLT19_IL/test_2019"
    ],
    # valid_datas=["/share/test/ztl/IL/IL_CHSTR/test"],
    select_data=["/share/test/ztl/IL/SynthMLT"],
    # select_data=["../../../test/ztl/IL/MLT17_IL/train_2017","../../../test/ztl/IL/MLT19_IL/train_2019"],
    batch_ratio="0.5-0.5",
    total_data_usage_ratio="1.0",
    NED=True,
    batch_size=256,
    num_iter=40000,
    val_interval=5000,
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
    eval_data="/share/test/ztl/IL/MLT17_IL/",
    eval_type="IL_STR",
    workers=4,
    batch_size=256,
    saved_model="saved_models/CRNN_real/best_score.pth", #saved_models/CRNN_real/best_score.pth
    log_multiple_test=None,
    NED=True,
    Aug=None,#|None|Blur|Crop|Rot|
    semi="None",

)
