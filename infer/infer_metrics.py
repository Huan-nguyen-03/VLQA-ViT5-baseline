import evaluate
from rouge_score import rouge_scorer

from tokenizer import tokenizer

bertscore = evaluate.load("bertscore")

def bert_score_calulate(predictions, references):
    predictions = [predictions]
    references = [references]
    results = bertscore.compute(
        predictions=predictions, references=references, model_type="bert-base-multilingual-cased"
    )
    return results['f1'][0]

def create_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def rouge_n(reference, candidate, n):
    reference_tokens = tokenizer.tokenize(reference)
    candidate_tokens = tokenizer.tokenize(candidate)
    
    reference_ngrams = create_ngrams(reference_tokens, n)
    candidate_ngrams = create_ngrams(candidate_tokens, n)
    
    intersection = set(reference_ngrams) & set(candidate_ngrams)
    intersection_count = 0
    
    for i in intersection:
        intersection_count += min(reference_ngrams.count(i), candidate_ngrams.count(i))
    
    reference_count = len(reference_ngrams)
    candidate_count = len(candidate_ngrams)
    
    precision = intersection_count / candidate_count if candidate_count > 0 else 0
    recall = intersection_count / reference_count if reference_count > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def rouge_l(reference, candidate):
    reference_tokens = tokenizer.tokenize(reference)
    candidate_tokens = tokenizer.tokenize(candidate)
    
    # Tính số lượng từ trùng lặp giữa câu tham chiếu và câu ứng viên
    intersection_count = len(set(reference_tokens) & set(candidate_tokens))
    
    # Tính số lượng từ riêng biệt trong câu tham chiếu
    reference_count = len(reference_tokens)
    
    # Tính số lượng từ riêng biệt trong câu ứng viên
    candidate_count = len(candidate_tokens)
    
    # Tính precision, recall và F1 score của Rouge-L
    precision = intersection_count / candidate_count if candidate_count > 0 else 0
    recall = intersection_count / reference_count if reference_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
