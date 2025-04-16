from transformers import GenerationConfig
import numpy as np

from model_config import *
from model import T5Model
from tokenizer import tokenizer
from infer.infer_metrics import *


generationConfig = GenerationConfig(
    temperature = 1
)

def generate_question(train_model, question, content_law):
    inputs_encoding =  tokenizer(
        question, content_law,
        add_special_tokens=True,
        max_length= INPUT_MAX_LEN,
        padding = 'max_length',
        truncation='only_second',
        return_attention_mask=True,
        return_tensors="pt"
        )

    inputs_encoding.to(DEVICE)
    generate_ids = train_model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = INPUT_MAX_LEN,
        num_beams = 4,
        num_return_sequences = 1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        generation_config = generationConfig,
        
        )

    preds = [
        tokenizer.decode(gen_id,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True)
        for gen_id in generate_ids
    ]

    return "".join(preds)


def infer(model, df_test):
    list_rouge1 = []
    list_rouge2 = []
    list_rouge3 = []
    list_rouge4 = []
    list_rougel = []
    list_em = []
    list_bert_score = []

    # Duyệt từng hàng và in ra các cột
    for index, row in df_test.iterrows():
        question = row["question"]
        content_Law = row["content_Law"]
        short_answer = row["answer"]
        
        answer_generated = generate_question(model, question, content_Law)
        
        # Sử dụng hàm cho ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-4
        rouge_1_precision, rouge_1_recall, rouge_1_f1 = rouge_n(short_answer, answer_generated, 1)
        rouge_2_precision, rouge_2_recall, rouge_2_f1 = rouge_n(short_answer, answer_generated, 2)
        rouge_3_precision, rouge_3_recall, rouge_3_f1 = rouge_n(short_answer, answer_generated, 3)
        rouge_4_precision, rouge_4_recall, rouge_4_f1 = rouge_n(short_answer, answer_generated, 4)
        rouge_l_precision, rouge_l_recall, rouge_l_f1 = rouge_l(short_answer, answer_generated)
        bert_sc = bert_score_calulate(answer_generated, short_answer)
        list_rouge1.append(rouge_1_f1)
        list_rouge2.append(rouge_2_f1)
        list_rouge3.append(rouge_3_f1)
        list_rouge4.append(rouge_4_f1)
        list_rougel.append(rouge_l_f1)
        list_bert_score.append(bert_sc)
        
        
        if short_answer == answer_generated:
            list_em.append(1)
        else:
            list_em.append(0)
        
        print(f"Question: {question}")
        print(f"Answer_generated: {answer_generated}")
        print(f"Labels: {short_answer}")

        print("-" * 50)  # In dấu gạch ngang để phân tách giữa các hàng

    mean_rouge1 = np.mean(list_rouge1)
    mean_rouge2 = np.mean(list_rouge2)
    mean_rouge3 = np.mean(list_rouge3)
    mean_rouge4 = np.mean(list_rouge4)
    mean_rougel = np.mean(list_rougel)
    mean_em = np.mean(list_em)
    mean_bert = np.mean(list_bert_score)

    print(f'Mean ROUGE-1: {mean_rouge1}')
    print(f'Mean ROUGE-2: {mean_rouge2}')
    print(f'Mean ROUGE-3: {mean_rouge3}')
    print(f'Mean ROUGE-4: {mean_rouge4}')
    print(f'Mean ROUGE-L: {mean_rougel}')
    print(f'Mean Exact Match (EM): {mean_em}')
    print(f'Mean BERT Score: {mean_bert}')