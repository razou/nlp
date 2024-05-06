import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load IMDb movie review dataset (or any other dataset)
# Assume the dataset is loaded into lists `reviews` and `labels` where labels are 0 for negative and 1 for positive sentiment
# Split the dataset into training and validation sets
reviews_train, reviews_val, labels_train, labels_val = train_test_split(reviews, labels, test_size=0.1, random_state=42)

max_length = 128
train_encodings = tokenizer(reviews_train, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(reviews_val, truncation=True, padding=True, max_length=max_length)

train_labels = torch.tensor(labels_train)
val_labels = torch.tensor(labels_val)

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              train_labels)

val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            val_labels)

batch_size = 16
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


if __name__ == "__main__":
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        predictions, true_labels = [], []
        for batch in val_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                total_val_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                predictions.append(logits)
                true_labels.append(label_ids)

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average validation loss: {avg_val_loss}")

    # Evaluate the model
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    predicted_class = np.argmax(predictions, axis=1)
    accuracy = np.sum(predicted_class == true_labels) / len(true_labels)
    print(f"Accuracy: {accuracy}")
