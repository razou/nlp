import random
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Union
import json
import logging
from datetime import datetime
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import evaluate


def save_config(config: Dict, path: str):
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for logging."""
    return " - ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])


def number_of_trainable_model_parameters(model) -> Tuple[int, int]:
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    output = {
        "num_trainable_parameters": trainable_model_params,
        "totam_num_parameters": all_model_params,
        "ratio_trainable_parameters": np.round(trainable_model_params / all_model_params, 3)
    }
    return output

def setup_logger(log_dir: str, log_level: str) -> logging.Logger:
    """Setup logger to output to file and console with configurable log level."""

    log_dir = os.path.join(log_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fh = logging.FileHandler(filename=os.path.join(log_dir, f'training_{timestamp}.log'), encoding='utf8', mode='a', delay=False)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def compute_rouge_score(reference_summarises: List[str], model_summaries: List[str]):
    rouge_metric = evaluate.load("rouge")
    return rouge_metric.compute(
            predictions=model_summaries,
            references=reference_summarises,
            use_aggregator=True,
            use_stemmer=True,
        )

@staticmethod
def generate_summaries(
    example: Dict[str, Any], 
    baseline_model: AutoModelForSeq2SeqLM, 
    peft_model: PeftModel,
    tokenizer: AutoTokenizer, 
    device: torch.device, 
    max_new_tokens: int
) -> Dict[str, str]:
    """Generate summaries for PEFT and baseline models."""
    prompt = f"Summarize the following conversation:\n\n{example['dialogue']}\n\nSummary:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    peft_summary = tokenizer.decode(
        peft_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)[0],
        skip_special_tokens=True
    )
    baseline_summary = tokenizer.decode(
        baseline_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)[0],
        skip_special_tokens=True
    )
    return {
        "peft_summary": peft_summary,
        "baseline_summary": baseline_summary,
    }
    

def generate_summaries_batch(
    batch: Dict[str, List[str]], 
    tokenizer: AutoTokenizer, 
    peft_model: PeftModel, 
    baseline_model: AutoModelForSeq2SeqLM, 
    device: torch.device, 
    max_new_tokens: int,
    dialogue_column: str
) -> Dict[str, List[str]]:
    """Batch processing for summary generation."""
    prompts = [
        f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:" for dialogue in batch[dialogue_column]
    ]
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    peft_outputs = peft_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    baseline_outputs = baseline_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)

    return {
        "peft_summary": tokenizer.batch_decode(peft_outputs, skip_special_tokens=True),
        "baseline_summary": tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)
    }

