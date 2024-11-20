import pathlib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr
import logging
from typing import Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationPredictor:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.label_names = self.model.config.id2label
        logger.info(f"Loaded model with labels: {self.label_names}")

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[Dict[str, float], str]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        raw_logits = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(raw_logits.logits, dim=-1)
        predictions = {self.label_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        predicted_label = max(predictions.items(), key=lambda x: x[1])[0]
        return predictions, predicted_label

    def create_gradio_interface(self):
        def predict_with_confidence(text: str) -> Tuple[Dict, str]:
            predictions, label = self.predict(text)
            confidence_str = "\n".join(f"{label}: {conf:.2%}"for label, conf in predictions.items())
            return confidence_str, label

        iface = gr.Interface(
            fn=predict_with_confidence,
            inputs=gr.Textbox(
                lines=5,
                placeholder="Enter text to classify...",
                label="Input Text"
            ),
            outputs=[
                gr.Textbox(label="Confidence Scores"),
                gr.Label(label="Predicted Class")
            ],
            title="Text Classification",
            description="Classify text using fine-tuned transformer architecture.",
            examples=[
                ["This movie was amazing! I loved every minute of it."],
                ["The worst film I've ever seen. Complete waste of time."],
                ["It was okay, nothing special but not terrible either."]
            ]
        )
        
        return iface

def main():
    config = {
        'model_path': 'outputs/model',
        'share': False,
        'server_port': 7860,
        'server_name': '0.0.0.0'
    }
    
    try:
        predictor = TextClassificationPredictor(config['model_path'])
        
        iface = predictor.create_gradio_interface()
        
        iface.launch(
            share=config['share'],
            server_port=config['server_port'],
            server_name=config['server_name']
        )
        
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()