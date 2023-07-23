common=dict(
    exp_name="CRNN_DER",  # Where to store logs and models
    il="der",  # joint_mix ｜ joint_loader | base | lwf | wa | ewc ｜ der  | mrn
    memory="random",  # None | random
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
    Transformation = "None",      #None TPS
    FeatureExtraction = "VGG",    #VGG ResNet
    SequenceModeling = "BiLSTM",  #None BiLSTM
    Prediction = "CTC",           #CTC Attn
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
    lan_list=["Chinese","Latin","Japanese", "Korean", "Arabic", "Bangla"],
    valid_datas=[
                 "../dataset/MLT17_IL/test_2017",
                 "../dataset/MLT19_IL/test_2019"
                 ],
    select_data=[
                 "../dataset/MLT17_IL/train_2017",
                 "../dataset/MLT19_IL/train_2019"
                 ],
    batch_ratio="0.5-0.5",
    total_data_usage_ratio="1.0",
    NED=True,
    batch_size=256,
    num_iter=10000,
    val_interval=5000,
    log_multiple_test=None,
    grad_clip=5,
)


