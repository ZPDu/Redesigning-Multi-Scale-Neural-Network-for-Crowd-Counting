from easydict import EasyDict as edict

# init
__C_SHHA = edict()

cfg_data = __C_SHHA

__C_SHHA.TRAIN_SIZE = (128,128) # 2D tuple or 1D scalar
__C_SHHA.DATA_PATH = ''

__C_SHHA.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

__C_SHHA.AUGMENT = 1 # the number of cropped patches within a single image
__C_SHHA.LOG_PARA = 100.

__C_SHHA.RESUME_MODEL = ''#model path
__C_SHHA.TRAIN_BATCH_SIZE = 1 #imgs

__C_SHHA.VAL_BATCH_SIZE = 1 # must be 1

__C_SHHA.EXP_PATH = ''
__C_SHHA.SEED = 640
__C_SHHA.MAX_EPOCH = 200
__C_SHHA.PRINT_FREQ = 10
__C_SHHA.EXP_NAME = 'HMoDE'
__C_SHHA.GPU_ID = [0]
