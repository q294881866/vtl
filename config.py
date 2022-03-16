from torchvision.transforms import transforms


class BaseConfig:
    # train conf
    EPOCH = 100
    TRAIN = 'train'
    TEST = 'test'
    # data config
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    NUM_CLASSES = 1
    NUM_FRAMES = 4
    BATCH_SIZE = 8
    FRAMES_STEP = 1
    ALL_DIM = 128
    base_lr = 1e-4
    image_based = True
    device_ids = [0, 1]
    shuffle = True
    IS_DISTRIBUTION = False
    denormalize = None
    HASH_BITS = 1024
    pretrained = ''
    type = ''
    train_h = True

    mean = [0.565, 0.556, 0.547]
    std = [0.232, 0.233, 0.234]
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    def __init__(self, mode, set_path, checkpoint, rank):
        self.mode = mode
        self.set_path = set_path
        self.checkpoint = checkpoint
        self.rank = rank


class DFTLConfig(BaseConfig):
    NUM_FRAMES = 8
    FRAMES_STEP = NUM_FRAMES // 2
    type = 'DFTL'

    def __init__(self, mode, set_path, checkpoint, rank):
        super().__init__(mode, set_path, checkpoint, rank)


class FFConfig(BaseConfig):
    NUM_FRAMES = 4
    FRAMES_STEP = NUM_FRAMES // 2
    choice_idx = 0
    type = 'FF'
    BATCH_SIZE = 2

    def __init__(self, mode, set_path, checkpoint, rank):
        super().__init__(mode, set_path, checkpoint, rank)


class SplicingConfig(BaseConfig):
    NUM_FRAMES = 4
    FRAMES_STEP = 1
    choice_idx = 0
    type = 'Splicing'
    BATCH_SIZE = 8

    def __init__(self, mode, set_path, checkpoint, rank):
        super().__init__(mode, set_path, checkpoint, rank)


class Davis2016Config(BaseConfig):
    NUM_FRAMES = 4
    FRAMES_STEP = 1
    type = 'Davis2016'
    BATCH_SIZE = 2

    def __init__(self, mode, set_path, checkpoint, rank):
        super().__init__(mode, set_path, checkpoint, rank)
