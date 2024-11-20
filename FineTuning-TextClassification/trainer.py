from transformers import (
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
from typing import Dict, Any
from metrics import MetricsCalculator
import logging


logger = logging.getLogger(__name__)

def setup_trainer(
    model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    config: Dict[str, Any]
) -> Trainer:
    """
    Setup the trainer with evaluation metrics
    """

    logger.info("Setting up training arguments")

    training_args = TrainingArguments(
        learning_rate=config['learning_rate'],
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        weight_decay=config['weight_decay'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config['metric_for_best_model'],
    )

    logger.debug(f"Training arguments: {training_args}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metrics_calculator = MetricsCalculator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=metrics_calculator.compute_metrics
    )
    
    logger.info(
        f"Trainer initialized with {len(train_dataset)} training samples and "
        f"{len(eval_dataset)} validation samples"
    )

    return trainer 