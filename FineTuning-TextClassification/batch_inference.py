import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from typing import List, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BatchInference:
    def __init__(self, model_path: str, batch_size: int = 32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.label_names = self.model.config.id2label

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predictions = [
            {self.label_names[i]: round(prob.item(), 3) for i, prob in enumerate(probs)}
            for probs in probabilities
        ]
        
        return predictions

    def predict_file(self, input_file: str, output_file: str, text_column: str):
        df = pd.read_csv(input_file, encoding='utf-8')
        texts = df[text_column].tolist()
        
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            predictions = self.predict_batch(batch_texts)
            all_predictions.extend(predictions)
        
        for label in self.label_names.values():
            df[f'prob_{label}'] = [pred[label] for pred in all_predictions]
        
        df['predicted_label'] = [
            max(pred.items(), key=lambda x: x[1])[0]
            for pred in all_predictions
        ]
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")

def main():
    config = {
        'model_path': 'outputs/model',
        'input_file': 'data/test.csv',
        'output_file': 'data/output_predictions.csv',
        'text_column': 'text',
        'batch_size': 32
    }
    
    try:
        predictor = BatchInference(
            model_path=config['model_path'],
            batch_size=config['batch_size']
        )
        
        predictor.predict_file(
            input_file=config['input_file'],
            output_file=config['output_file'],
            text_column=config['text_column']
        )
        
    except Exception as e:
        logger.error(f"Error in batch inference: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 