import argparse
import copy
import os
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from utils import compute_rouge_score
import logging


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
logger = logging.getLogger(__name__)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="knkarthick/dialogsum", type=str)
    parser.add_argument("--label_key", default="summary", type=str)
    parser.add_argument("--dialogue_text", default="dialogue", type=str)
    parser.add_argument("--model_dir", default="peft_model_aws_s3", type=str)
    parser.add_argument("--max_length", default=512, type=int, help="Max input length for tokenization")
    parser.add_argument("--max_new_tokens", default=128, type=int, help="Max tokens for generation")
    parser.add_argument("--test_sample_size", default=100, type=int)
    parser.add_argument("--log_level", default="INFO", type=str, help="Set logging level")
    parsed_args = parser.parse_args()
    return parsed_args

def main():
    args = _parse_args()

    log_level = getattr(logging, args.log_level.upper().strip(), None)
    if not isinstance(log_level, int):
        raise ValueError("Wrong log lovel value")
    logger.setLevel(log_level)

    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory '{args.model_dir}' does not exist.")
        return

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    peft_config = PeftConfig.from_pretrained(args.model_dir)

    logger.info(f"PEFT Configuration: {peft_config}")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=peft_config.base_model_name_or_path, 
        device_map='auto',
        torch_dtype='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path)

    logger.info('Loading PEFT model')
    peft_model = PeftModel.from_pretrained(
        model=base_model, 
        model_id=args.model_dir, 
        is_trainable=False
    )

    peft_model.to(device)

    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    
    logger.info("Loading and sampling test dataset")
    sampled_test_dataset = load_dataset(args.dataset_name, split='test')
    

    if args.test_sample_size > 0:
        if len(sampled_test_dataset) > args.test_sample_size:
            sampled_test_dataset = sampled_test_dataset.shuffle().select(range(args.test_sample_size))
        
    peft_model_summaries, baseline_model_summaries, human_baseline_summaries = [], [], []

    for i, example in tqdm(enumerate(sampled_test_dataset), desc="Generating summaries"):
        dialogue = example[args.dialogue_text]
        summary = example[args.label_key]

        human_baseline_summaries.append(summary)

        prefix = "Summarize the following conversation:\n\n"
        suffix = "\n\nSummary:"
        prompt = prefix + dialogue + suffix

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = inputs["input_ids"].to(device)

        try:
            peft_outputs = peft_model.generate(input_ids=input_ids, generation_config=generation_config)
            peft_outputs_decoded = tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
            peft_model_summaries.append(peft_outputs_decoded)

            with peft_model.disable_adapter():
                baseline_outputs = peft_model.generate(input_ids=input_ids, generation_config=generation_config)
                baseline_outputs_decoded = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
                baseline_model_summaries.append(baseline_outputs_decoded)

            logger.info(f"\n=>Dialogue: #{i + 1}\n{dialogue}")
            logger.info(f" => Baseline Model Summary: {baseline_outputs_decoded}")
            logger.info(f" => PEFT Model Summary: {peft_outputs_decoded}")
            logger.info(f" => Human Baseline Summary: {summary}")
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"Error occured durring text generation: {e}")
            continue

    if not peft_model_summaries or not baseline_model_summaries:
        logger.error("No valid summaries were generated.")
        return
    
    peft_model_scores = compute_rouge_score(
        reference_summarises=human_baseline_summaries, 
        model_summaries=peft_model_summaries
    )
    baseline_model_scores = compute_rouge_score(
        reference_summarises=human_baseline_summaries, 
        model_summaries=baseline_model_summaries
    )

    logger.info("\nPEFT Model Scores:")
    for metric, score in peft_model_scores.items():
        logger.info(f"{metric}: {score:.4f}")

    logger.info("\nBaseline Model Scores:")
    for metric, score in baseline_model_scores.items():
        logger.info(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()

