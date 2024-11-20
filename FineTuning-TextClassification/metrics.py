import evaluate
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    def __init__(self):
        self.metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute metrics using combined evaluate approach
        """
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        try:
            results = self.metrics.compute(predictions=predictions, references=labels)
            logger.debug(f"Computed metrics: {results}")
            return results
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise 