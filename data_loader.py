import pytorch_lightning as pl

from tokenizer import tokenizer
from model_config import *
from dataset import T5Dataset

class T5DataLoad(pl.LightningDataModule):
    
    def __init__(self,df_train,df_test):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUTPUT_MAX_LEN
    
    def setup(self, stage=None):
        # Sửa ở đây
        self.train_data = T5Dataset(
            question = self.df_train.question.values,
            content_Law = self.df_train.content_Law.values,
            answer = self.df_train.answer.values
#             answer = self.df_train.short_answer.values
        )
        
        self.valid_data = T5Dataset(
            question = self.df_test.question.values,
            content_Law = self.df_test.content_Law.values,
            answer = self.df_test.answer.values
#             answer = self.df_test.short_answer.values
        )
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
         self.train_data,
         batch_size= TRAIN_BATCH_SIZE,
         shuffle=True, 
         num_workers=2
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
        self.valid_data,
        batch_size= VAL_BATCH_SIZE,
        num_workers = 2
        )
