import argparse
from typing import Tuple, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch
import logging
import gradio as gr

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(
            self, 
            model: Union[PeftModel, AutoModelForSeq2SeqLM], 
            tokenizer: AutoTokenizer, 
            device: torch.device = "cpu",
            max_new_token: int = 128
            ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens: int  = max_new_token
        self.device = device
        
        
    def generate_summary(self, dialogue: str) -> str:
        """Generate summary for a dialogue using PEFT model."""
        
        prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary: """
        
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=self.max_new_tokens)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def batch_generate_summaries(self, dialogues: list) -> list:
        """Generate summaries for a batch of dialogues."""
        return [self.generate_summary(d) for d in dialogues]


def load_peft_model(model_path: str) -> Tuple[PeftModel, AutoTokenizer]:
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            peft_config.base_model_name_or_path, 
            torch_dtype='auto', 
            device_map='auto',
        )
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, use_fast=True, device_map="auto")
        peft_model = PeftModel.from_pretrained(
            base_model, 
            model_path, 
            torch_dtype='auto', 
            is_trainable=False
        )
        return peft_model, tokenizer
    except Exception as e:
        logger.error(f"Unable to load Peft model due to {str(e)}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="peft_model", type=str)
    parsed_args = parser.parse_args()
    return parsed_args

def main() -> None:

    args = _parse_args()
    model_dir = args.model_dir
    device = "cpu"

    logger.info(f'Load Peft Model from {model_dir} path')
    peft_model_inference, tokenizer = load_peft_model(model_path=model_dir)
    peft_model_inference.to(device) 
    peft_model_inference.eval()

    summarizer = Summarizer(model=peft_model_inference, tokenizer=tokenizer)

    iface = gr.Interface(
        fn=summarizer.generate_summary,
        inputs=gr.Textbox(
            lines=5,
            placeholder="Enter dialogue to summarize",
            label="Input dialogue"
        ),
        outputs=gr.Textbox(
            lines=5,
            label="Summary"
        ),
        
        title="Dialogue summarizer",
        description="Summarize dialogue using Peft fine-tuned Flan-T5 model",
    )

    iface.launch()

  

if __name__ == "__main__":
    main()
