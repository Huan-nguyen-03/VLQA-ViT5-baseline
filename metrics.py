from collections import Counter

def calculate_rouge(hypothesis, reference):
    # Chuyển danh sách từ thành danh sách bigram
    hypothesis_bigrams = [(hypothesis[i], hypothesis[i + 1]) for i in range(len(hypothesis) - 1)]
    reference_bigrams = [(reference[i], reference[i + 1]) for i in range(len(reference) - 1)]
    
    # Đếm số lần xuất hiện của từng bigram trong danh sách tham chiếu
    reference_bigram_counts = Counter(reference_bigrams)
    
    # Tính số bigram chung
    common_bigrams = set(hypothesis_bigrams) & set(reference_bigrams)
    common_bigram_count = sum(reference_bigram_counts[bigram] for bigram in common_bigrams)
    
    # Tính tỷ lệ ROUGE-2
    precision = common_bigram_count / len(hypothesis_bigrams)
    recall = common_bigram_count / len(reference_bigrams)
    
    # Tính F1 score (trung bình điều hòa của precision và recall)
    if precision + recall == 0:
        rouge_2_f1 = 0.0
    else:
        rouge_2_f1 = (2.0 * precision * recall) / (precision + recall)
    
    return rouge_2_f1
