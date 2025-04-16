from tokenizer import tokenizer
from model_config import *

class T5Dataset:
    
  def __init__(self,question,content_Law,answer):   
    
    self.question = question
    self.answer = answer
    self.content_Law = content_Law
    self.tokenizer = tokenizer
    self.input_max_len = INPUT_MAX_LEN
    self.output_max_len = OUTPUT_MAX_LEN
  
  def __len__(self):                      # This method retrives the number of item from the dataset
    return len(self.question)

  def __getitem__(self,item):             # This method retrieves the item at the specified index item. 

    question = str(self.question[item])
    content_Law = str(self.content_Law[item])
    answer = str(self.answer[item])

    input_tokenize = self.tokenizer(      
            question, content_Law,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding = 'max_length',
            truncation = 'only_second',
            return_attention_mask=True,
            return_tensors="pt"
        )
    output_tokenize = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.output_max_len,
            padding = 'max_length',
            truncation = 'only_second',
            return_attention_mask=True,
            return_tensors="pt"
            
        )
    

    input_ids = input_tokenize["input_ids"].flatten()
    attention_mask = input_tokenize["attention_mask"].flatten()
    labels = output_tokenize['input_ids'].flatten()

    out = {
            'question':question,      
            'content_Law': content_Law,
            'answer':answer,
            'input_ids': input_ids,
            'attention_mask':attention_mask,
            'target':labels
        }
        
    return out      
