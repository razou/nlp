# Text Classification with BERT

This project demonstrates how to fine-tune a pre-trained transformer (e.g., BERT) model for text classification task using the IMDB movie reviews dataset. It mainly uses the Hugging Face ecosystem of libraries.

## Prerequisites

- Python version >= 3.9.2
- Required libraries listed in requirements.txt

## Components

### 1. Data Loading and Processing

- **Hugging Face Datasets**
  - Access high-quality datasets from [Hugging Face Hub](https://huggingface.co/datasets)
  - Support for various NLP tasks including classification, sentiment analysis, and QA
  - Efficient data processing using the [`datasets`](https://huggingface.co/docs/datasets/package_reference/main_classes) library
  - Built-in preprocessing and data augmentation capabilities

### 2. Model Training

- **Fine-tuning Transformer Models**
  - Using pre-trained BERT for text classification
  - Leveraging the [Trainer API](https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/trainer#transformers.Trainer) for optimized PyTorch training
  - Browse available models at [Text Classification Models](https://huggingface.co/tasks/text-classification)

### 3. Evaluation

- **Model Assessment**
  - Comprehensive evaluation using the [`evaluate`](https://huggingface.co/docs/evaluate/index) library
  - Standard metrics implementation via [API Classes](https://huggingface.co/docs/evaluate/package_reference/main_classes)
  - Performance tracking and visualization

## How to use

- **Training**
  1. Clone the repository
  2. Install the required libraries
  3. Run the `train.py` script
  4. The model will be saved to the `outputs` folder

- **Inference**
  1. Run the `inference.py` script
  2. The model will be loaded from the `outputs` folder
  3. The Gradio interface will be launched

- **Batch Inference**
  1. Run the `batch_inference.py` script
  2. The model will be loaded from the `outputs` folder
  3. The predictions will be saved to the `data` folder

