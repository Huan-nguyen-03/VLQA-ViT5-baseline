import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_MAX_LEN = 1024 #input length
OUTPUT_MAX_LEN = 1024 # output length
TRAIN_BATCH_SIZE = 1 # batch size of training
VAL_BATCH_SIZE = 1 # batch size for validation
EPOCHS = 6 # number of epoch
MODEL_NAME = "VietAI/vit5-base"

TRAIN_DATA_PATH = "./VLQA_data/train_data_example.json"
TEST_DATA_PATH = "./VLQA_data/test_data_example.json"
LAW_DATA_PATH = "./VLQA_data/law_data_example.json"

OUTPUT_DIR = "./output"
