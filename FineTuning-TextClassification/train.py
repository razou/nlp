from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, load_dataset, ClassLabel
from trainer import setup_trainer
import logging
from typing import Dict, Any
import torch
import psutil


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_data(dataset: Dataset, tokenizer, config: Dict[str, Any]) -> Dataset:
    """
    Process a dataset with tokenization
    """
    logger.info(f"Processing dataset with {len(dataset)} samples")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config['max_length']
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=config['batch_size'],
        num_proc=config['num_proc'],
        desc="Tokenizing datasets"
    )
    logger.debug(f"Dataset processed. Features: {tokenized_dataset.features}")
    return tokenized_dataset

def main():
    
    cpu_count = psutil.cpu_count(logical=False)
    num_proc = max(1, cpu_count - 1)

    config = {
        'model_name': 'bert-base-uncased',
        'dataset_name': 'jahjinx/IMDb_movie_reviews',
        'output_dir': './outputs',
        'learning_rate': 5e-5,
        'num_epochs': 3,
        'batch_size': 32,
        'weight_decay': 0.01,
        'metric_for_best_model': 'f1',
        'max_length': 128,
        'sample_size': 2000,
        'label_mapping': {'negative': 0, 'positive': 1},
        'num_proc': num_proc,
        'gradient_accumulation_steps': 4,
        'fp16': True
    }
    
    logger.debug(f"Configuration: {config}")

    logger.info(f"Label mapping: {config['label_mapping']}")
    num_labels = len(config['label_mapping'])
    label_names = list(config['label_mapping'].keys())
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = config['label_mapping']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(num_proc)
    else:
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    model = model.to(device)

    logger.info(f"Model architecture: {model.config.architectures[0]}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Loading datasets")

    if config.get('sample_size'):
        dataset = load_dataset(
            config['dataset_name'], 
            split=f"train[:{config['sample_size']}]"
            )
        logger.info(f"Loaded {config['sample_size']} samples from dataset")
    else:
        dataset = load_dataset(config['dataset_name'], split="train")
        logger.info(f"Loaded full dataset with {len(dataset)} samples")

    features = dataset.features.copy()
    features['label'] = ClassLabel(num_classes=num_labels, names=label_names)
    dataset = dataset.cast(features)
    
    dataset = dataset.shuffle(seed=42)
    dataset_splits = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=1)
    
    logger.info(f"Train dataset shape: {dataset_splits['train'].shape}")
    logger.info(f"Eval dataset shape: {dataset_splits['test'].shape}")
    
    logger.debug(f"Sample from train dataset: {dataset_splits['train'][0]}")

    logger.info("Processing datasets")
    train_dataset = process_data(dataset_splits['train'], tokenizer, config)
    eval_dataset = process_data(dataset_splits['test'], tokenizer, config)

    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )

    logger.info("Starting training")
    train_results = trainer.train()
    logger.info(f"Training completed. Results: {train_results}")

    logger.info("Saving model")
    trainer.save_model(f"{config['output_dir']}/model")
    tokenizer.save_pretrained(f"{config['output_dir']}/tokenizer")

    logger.info("Running final evaluation")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    return train_results, eval_results

if __name__ == "__main__":
    main()