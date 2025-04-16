from sklearn.metrics import f1_score
from transformers import T5ForConditionalGeneration
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim import AdamW


from tokenizer import tokenizer
from model_config import *
from metrics import calculate_rouge

class T5Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)
        self.epoch = -1
        self.validation_loss = []
        self.validation_f1 = []
        self.validation_em = []
        self.validation_rouge = []
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        output = self.model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        labels=labels
        )
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["target"]
        loss, logits = self(input_ids , attention_mask, labels)
   
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["target"]
        loss, logits = self(input_ids, attention_mask, labels)
        
        
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)

        predicted_class = predicted_class.cpu().numpy()
        labels = labels.cpu().numpy()
        
        ## loại bỏ các vị trí cùng là 0
        ## cái trên thì dùng để decode, cái này dùng để validate
        pairs = [[(a, b) for a, b in zip(predicted_class[i], labels[i]) if a != 0 or b != 0] for i in range(TRAIN_BATCH_SIZE)]
        predicted_class_filtered = []
        labels_filtered = []
        for i in range(TRAIN_BATCH_SIZE):
            p_filtered, l_filtered = zip(*pairs[i])
            predicted_class_filtered.append(p_filtered)
            labels_filtered.append(l_filtered)
        

        f1 = 0
        em = 0
        rouge = 0
        for i in range(TRAIN_BATCH_SIZE):
            p = predicted_class_filtered[i]
            l = labels_filtered[i]
            
            if np.array_equal(l, p):
                em = em + 1
            f1 = f1 + f1_score(l, p, average='micro')   
            rouge = calculate_rouge(p, l)

        em = em / TRAIN_BATCH_SIZE
        f1 = f1 / TRAIN_BATCH_SIZE
        rouge = rouge / TRAIN_BATCH_SIZE
#         print("f1: ", f1)
#         print("rouge: ", rouge)
#         print('-' * 20)
        self.validation_loss.append(loss)
        self.validation_f1.append(f1)
        self.validation_em.append(em)
        self.validation_rouge.append(rouge)
        
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)
        self.log("val_em", em, prog_bar=True, logger=True)
        self.log("val_rouge", rouge, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_f1': f1, 'val_em': em, "val_rouge": rouge}
    
    
    def on_validation_epoch_end(self):
        all_loss = self.validation_loss
        all_f1 = self.validation_f1
        all_em = self.validation_em
        all_rouge = self.validation_rouge

        print("epoch: ", self.epoch)
        self.epoch = self.epoch + 1
        print()

        losses_tensor = torch.stack(all_loss)
        average_loss = torch.mean(losses_tensor)
        average_f1 = sum(all_f1) / len(all_f1)
        average_em = sum(all_em) / len(all_em)
        average_rouge = sum(all_rouge) / len(all_rouge)
        print("average_loss: ", average_loss)
        print("average_f1: ", average_f1)
        print("average_em: ", average_em)
        print("average_rouge: ", average_rouge)
        print("-" * 20)

        self.validation_loss.clear()
        self.validation_f1.clear()
        self.validation_em.clear()
        self.validation_rouge.clear()
        

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)