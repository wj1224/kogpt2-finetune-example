class GeneralConfig:
    train_path = 'train.csv'
    test_path = 'val.csv'

    dialect = '<unused0>'
    standard = '<unused1>'
    bos = '<s>'
    eos = '</s>'
    mask = '<unused2>'
    pad = '<pad>'
    unk = '<unk>'
    max_len = 128

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TrainerConfig:
    amp = True
    dp = False
    device = '0'
    save_dir = "checkpoints"
    
    batch_size = 16
    num_workers = 0
    shuffle=True

    lr = 5e-5
    epochs = 5
    warmup_ratio = 0.1
    grad_norm_clip = 1.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
