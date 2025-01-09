import gc
import psutil
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
from typing import Dict, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self, 
        output_dir: str,
        log_dir: str,
        base_model_path_or_name: str,
        label_key: str = "summary",
        dialogue_text: str = "dialogue",
        metric: str = "rouge", 
        batch_size: int = 16,
        max_length_source: int = 512,
        max_length_target: int = 128,
    ) -> None:
        
        self.label_key = label_key
        self.dialogue_text = dialogue_text
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.max_length_source = max_length_source
        self.max_length_target = max_length_target
        self.batch_size =  batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path_or_name, use_fast=True)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path_or_name, torch_dtype="auto", device_map="auto")
        self.metric = evaluate.load(metric)
    
    @staticmethod
    def inspect_dataset(dataset: Union[Dataset, DatasetDict]) -> None:
        """Utility to inspect dataset structure."""
        if isinstance(dataset, dict):
            for split, data in dataset.items():
                logger.info(f"{split.capitalize()} Dataset: {len(data)} samples")
                logger.info(f"Columns: {data.column_names}")
        else:
            logger.info(f"Dataset size: {len(dataset)} samples")
            logger.info(f"Columns: {dataset.column_names}")

    def data_preprocessing_batch(
            self, 
            data_set: Dataset,
            num_proc: int,
            prefix: str = "Summarize the following conversation:\n\n",
            suffix: str = "\n\nSummary:", 
        ) -> Dataset:

        def tokenizer_func(sample):
            """Data tokenizer: Iterate over input data and tokenize per batch"""
            
            prompt = [prefix + dialogue + suffix for dialogue in sample[self.dialogue_text]]
            input_encoding  = self.tokenizer(
                prompt, 
                padding="max_length",
                max_length=self.max_length_source,
                truncation=True, 
                return_tensors="pt"
            )
            label_encoding = self.tokenizer(
                sample[self.label_key], 
                padding="max_length", 
                max_length=self.max_length_target,
                truncation=True, 
                return_tensors="pt"
            )
            sample["input_ids"] = input_encoding.input_ids
            sample["labels"] = label_encoding.input_ids
            return sample
        
        logger.info("Inspecting dataset before preprocessing...")
        self.inspect_dataset(dataset=data_set)

        tokenized_dataset = data_set.map(
            tokenizer_func,
            batched=True,
            num_proc=num_proc,
            remove_columns=data_set['train'].column_names,
            load_from_cache_file=False,
            desc="Tokenizer"
        )
        logger.info("Dataset preprocessing completed.")
        return tokenized_dataset

    def compute_metrics(self, eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            logits = predictions[0]
            token_ids = np.argmax(logits, axis=-1)
        else:
            token_ids = predictions
        
        decoded_preds = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer does not have a pad_token_id. Ensure the tokenizer is correctly initialized.")

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: np.round(v, 4) for k, v in result.items()}
    
    @staticmethod
    def get_peft_config(config: Optional[Dict[str, Any]] = None) -> LoraConfig:
        """Create LoRA config for PEFT fine-tuning."""

        default_config = {
                "r": 32,
                "lora_alpha": 32,
                "target_modules": ["q", "v"],
                "lora_dropout": 0.05,
                "inference_mode": False
        }

        if config is not None:  
            default_config.update(config)

        return LoraConfig(
            r=default_config["r"],
            lora_alpha=default_config["lora_alpha"],
            target_modules=default_config["target_modules"],
            lora_dropout=default_config["lora_dropout"],
            bias="none",
            inference_mode=default_config['inference_mode'],
            task_type=TaskType.SEQ_2_SEQ_LM
        )

    @staticmethod
    def verify_requires_grad(model) -> None:
        """
        - Ensures that only the intended parameters are being updated during training.
        - Helps catch configuration errors where non-LoRA parameters might inadvertently be set to requires_grad=True or vice versa
        """
        logger.info("Verifying requires_grad settings...")
        for name, param in model.named_parameters():
            if "lora" in name and not param.requires_grad:
                logger.error(f"Layer {name} in LoRA is expected to have requires_grad=True but is False.")
            elif "lora" not in name and param.requires_grad:
                logger.error(f"Layer {name} outside LoRA is expected to have requires_grad=False but is True.")
        logger.info("Verification of requires_grad settings completed.")

    @staticmethod
    def verify_lora_parameters(model, stage: str) -> None:
        """Logs the mean and standard deviation of parameters in the LoRA layers."""
        logger.info(f"Lora parameter values -- {stage}:")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                logger.info(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

    def train(
        self, 
        dataset: Union[Dataset, DatasetDict],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        
        cpu_count = psutil.cpu_count(logical=False)
        num_proc = min(cpu_count - 1, max(4, cpu_count // 2))

        default_config = {
            "overwrite_output_dir": True,
            "learning_rate": 1e-4,
            "num_train_epochs": 5,
            "weight_decay": 0.01,
            "max_steps": -1,
            "eval_strategy": "steps",
            "logging_strategy": "steps",
            "logging_steps": 10,
            "log_level": "info",
            "auto_find_batch_size": True
        }
   
        if config is not None:
            default_config.update(config)

        logger.info("Starting training with parameters:")
        logger.info(f"Parameters: {default_config}")

        logger.info("Preprocessing dataset...")    
        tokenized_dataset = self.data_preprocessing_batch(dataset, num_proc)

        if isinstance(tokenized_dataset, dict):
            logger.info(f"Tokenized Training Set: {len(tokenized_dataset['train'])} samples")
            if "validation" in tokenized_dataset:
                logger.info(f"Tokenized Validation Set: {len(tokenized_dataset['validation'])} samples")
            if "test" in tokenized_dataset:
                logger.info(f"Tokenized Test Set: {len(tokenized_dataset['test'])} samples")
        else:
            logger.info(f"Tokenized Dataset: {len(tokenized_dataset)} samples")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=default_config["overwrite_output_dir"],
            num_train_epochs=default_config['num_train_epochs'],
            #max_steps=default_config["max_steps"],
            logging_steps=default_config['logging_steps'],
            auto_find_batch_size=default_config["auto_find_batch_size"],
            learning_rate=default_config["learning_rate"],
            #optim="adamw_torch",
            push_to_hub=False,
            weight_decay=default_config["weight_decay"],
            #eval_strategy=default_config["eval_strategy"],
            logging_strategy=default_config["logging_strategy"],
            save_strategy=default_config["save_strategy"],
            #save_steps=default_config["save_steps"],
            logging_dir=self.log_dir,
            log_level=default_config["log_level"]
        )

        logger.info('Create Peft model')
        peft_config = self.get_peft_config()
        peft_model = get_peft_model(model=self.base_model, peft_config=peft_config)
        logger.info(f"\n{peft_model.print_trainable_parameters()}\n")

        logger.info("Verify and log parameter settings")
        self.verify_requires_grad(peft_model)
        self.verify_lora_parameters(peft_model, "Before training")

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"] if isinstance(tokenized_dataset, dict) else tokenized_dataset,
            eval_dataset=tokenized_dataset.get("validation", None) if isinstance(tokenized_dataset, dict) else None,
            compute_metrics=self.compute_metrics
        )

        logger.info(f"Starting training for {default_config['num_train_epochs']} epochs...")
        trainer.train()

        logger.info("Log parameter values after training")
        self.verify_lora_parameters(peft_model, "After training")

        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        print("eval_results: ", eval_results)

        logger.info(f"Saving model to {self.output_dir}")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Training completed successfully")

        logger.info(f"Training loss history: {trainer.state.log_history}")

        logger.info("Clear memory")
        del self.base_model, self.tokenizer, peft_model, tokenized_dataset
        gc.collect()