import argparse
import logging
import os
from datasets import load_dataset
from trainer import ModelTrainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def _parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_name", default="knkarthick/dialogsum", type=str, help="The dataset (on hugginface) name or path")
        parser.add_argument("--label_key", default="summary", type=str, help="Label column name within input dataset")
        parser.add_argument("--dialogue_text", default="dialogue", type=str, help="Dialogue column name wirhing input dataset")
        parser.add_argument("--model_name", default="google/flan-t5-base", type=str, help="Base model name or path")

        parser.add_argument("--output_dir", default="peft_model", type=str, 
                            help="The output directory where the model predictions and checkpoints will be written")
        parser.add_argument("--overwrite_output_dir", default=True, help="Whether or not to owerwrite outpudir")

        parser.add_argument("--learning_rate", default=1e-5, type=float, help="Initial learning rate for AdamW")
        parser.add_argument("--weight_decay", default=0.01, type=float, help="Weigh decay")

        parser.add_argument("--num_train_epochs", default=1, type=int, help="The number of epochs")
        parser.add_argument("--max_steps", default=100, type=int, help="It will override any value given in num_train_epochs")

        parser.add_argument("--eval_strategy", default="steps", type=str, choices= ["no", "steps", "epoch"], help="Evaluation strategy")

        parser.add_argument("--logging_strategy", default="steps", type=str, choices= ["no", "steps", "epoch"], help="Logging strategy")
        parser.add_argument("--logging_steps", default=1, type=int)

        parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch", "best"], 
                            help="The checkpoint save strategy to adopt during training")
        parser.add_argument("--save_steps", type=int, default=10, help="#Updates steps before two checkpoint saves if save_strategy='steps'")

        parser.add_argument("--data_filter_ratio", default=0
                            , type=int, help="The amount by which to devide the input dataset for quick train/debug")

        parser.add_argument("--log_level", default="INFO", type=str, help="Set logging level")
        parsed_args = parser.parse_args()
        return parsed_args


def main() -> None:
    
    args = _parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    label_key = args.label_key
    dialogue_text = args.dialogue_text
    log_dir = f"logs-dir-{str(os.path.split(args.output_dir)[-1]).replace('_', '-')}"
    output_dir = args.output_dir
    data_filter_ratio = args.data_filter_ratio

    log_level = getattr(logging, args.log_level.upper().strip(), None)
    if not isinstance(log_level, int):
        raise ValueError("Wrong log lovel value")
    logger.setLevel(log_level)


    logger.info(f"Loading '{dataset_name}' dataset...")
    dataset = load_dataset(dataset_name)
    logger.info(f"Dataset info: {dataset}")
    if data_filter_ratio > 0:
        dataset = dataset.filter(lambda example, index: index % data_filter_ratio == 0, with_indices=True)
    
    trainer = ModelTrainer(
        output_dir=output_dir, 
        log_dir=log_dir,
        label_key=label_key,
        dialogue_text=dialogue_text,
        base_model_path_or_name=model_name
    )

    train_config = {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            #"max_steps": args.max_steps,
            "eval_strategy": args.eval_strategy,
            "logging_strategy": args.logging_strategy,
            "logging_steps": args.logging_steps,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "overwrite_output_dir": args.overwrite_output_dir,
            "auto_find_batch_size":True
        }
    
    trainer.train(dataset=dataset, config=train_config)

if __name__ == "__main__":
    main()