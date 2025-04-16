from transformers import GenerationConfig

from model_config import *
from model import T5Model
from tokenizer import tokenizer
from infer.infer_utils import infer
from vlqa_data import get_vlqa_data
from pathlib import Path


if __name__ == "__main__":
    # Load model
    checkpoint_path = Path(OUTPUT_DIR) / "best-model-loss.ckpt"
    trained_model = T5Model.load_from_checkpoint(checkpoint_path)
    trained_model.freeze()
    trained_model.to(DEVICE)
    
    # Load data
    df_train, df_test = get_vlqa_data()
    
    infer(trained_model, df_test)

