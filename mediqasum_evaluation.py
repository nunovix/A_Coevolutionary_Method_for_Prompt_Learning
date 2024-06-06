import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import evaluate
import numpy as np
import nltk
from typing import List, Tuple
import torch
import tensorflow as tf

def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

nltk.download('punkt')

def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text

def postprocess_text(preds, labels) -> Tuple[List[str], List[str]]:
    preds = [sanitize_text(pred) for pred in preds]
    labels = [sanitize_text(label) for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

# if only_rouge ==True, only rouge 1 is returned
def evaluate_texts(predictions: List[str], references: List[str], only_rouge = True,
                   batch_size: int = 1, cache_dir: str = "./cache"):
    
    batch_size = len(predictions)
    # Postprocess texts
    predictions, references = postprocess_text(predictions, references)
    
    # Prepare to collect batch results
    all_rouge1, all_bertscore_f1s, all_bleurt_scores = [], [], []

    # Evaluate with ROUGE
    rouge = evaluate.load("rouge", cache_dir=cache_dir)

    for i in range(0, len(predictions), batch_size):
        print(f"rougen\n\n")
        batch_predictions = predictions[i:i + batch_size]
        batch_references = references[i:i + batch_size]
        rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

        print(f"batch_predictions-->{batch_predictions}")
        print(f"batch_references-->{batch_references}")
        print(f"rouge_results-->{rouge_results}")

        all_rouge1.append(rouge_results["rouge1"])

    if only_rouge == True:
        return np.mean(all_rouge1).item()*100

    # Evaluate with BERTScore
    bertscore = evaluate.load("bertscore", cache_dir=cache_dir)
    for i in range(0, len(predictions), batch_size):
        #print(f"bert\n\n")
        batch_predictions = predictions[i:i + batch_size]
        batch_references = references[i:i + batch_size]
        bertscore_result = bertscore.compute(predictions=batch_predictions, references=batch_references, model_type="bert-base-uncased", lang="en", rescale_with_baseline=True)
        bertscore_f1 = np.mean(bertscore_result["f1"])
        all_bertscore_f1s.append(bertscore_f1)

    del bertscore  # Unload model after use
    torch.cuda.empty_cache()
    bleurt = evaluate.load("bleurt", "BLEURT-20-D12", cache_dir=cache_dir)
    
    for i in range(0, len(predictions), batch_size):

        batch_predictions = predictions[i:i + batch_size]
        batch_references = references[i:i + batch_size]

        bleurt_result = bleurt.compute(predictions=batch_predictions, references=batch_references)
        bleurt_score = np.mean(bleurt_result["scores"])
        all_bleurt_scores.append(bleurt_score)

    del bleurt  # Unload model after use
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    # Average results from all batches
    results = {
        "rouge1_avg": np.mean(all_rouge1),
        "bertscore_f1": np.mean(all_bertscore_f1s),
        "bleurt_score": np.mean(all_bleurt_scores),
        "ensemble_gen_score": np.mean([np.mean(all_rouge1), np.mean(all_bertscore_f1s), np.mean(all_bleurt_scores)])
    }

    rounded_results = {k: round(v * 100, 4) for k, v in results.items()}

    return rounded_results