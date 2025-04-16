from transformers import T5Tokenizer, T5ForConditionalGeneration  
from model_config import *


tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=INPUT_MAX_LEN)
