import pandas as pd
import numpy as np
import re
import string
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    RobertaModel
)
from datasets import Dataset, DatasetDict
import torch.nn as nn

# ----------------------------
# TEXT PREPROCESSING FUNCTION
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# LOAD & CLEAN DATA
# ----------------------------
df = pd.read_csv("Balanced_CBH_Dataset_Preprocessed.csv")
df.dropna(subset=['Text', 'Types'], inplace=True)
df['Text'] = df['Text'].apply(clean_text)

label_counts = df['Types'].value_counts()
df = df[df['Types'].isin(label_counts[label_counts >= 2].index)].reset_index(drop=True)

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['Types'])

# ----------------------------
# TRAIN / VAL / TEST SPLIT
# ----------------------------
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df['Text'], df['label_encoded'], test_size=0.2, stratify=df['label_encoded'], random_state=42
)

train_val_df = pd.DataFrame({'Text': train_val_texts, 'label_encoded': train_val_labels})
val_counts = train_val_df['label_encoded'].value_counts()
train_val_df = train_val_df[train_val_df['label_encoded'].isin(val_counts[val_counts >= 2].index)].reset_index(drop=True)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_df['Text'], train_val_df['label_encoded'],
    test_size=0.125, stratify=train_val_df['label_encoded'], random_state=42
)

# ----------------------------
# TOKENIZATION & DATASET
# ----------------------------
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(batch['Text'], truncation=True, padding=True, max_length=128)

train_ds = Dataset.from_dict({'Text': train_texts, 'label': train_labels})
val_ds = Dataset.from_dict({'Text': val_texts, 'label': val_labels})
test_ds = Dataset.from_dict({'Text': test_texts, 'label': test_labels})

dataset = DatasetDict({
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
}).map(tokenize, batched=True)

# ----------------------------
# HYBRID MODEL: RoBERTa + LSTM
# ----------------------------
class RobertaLSTMClassifier(nn.Module):
    def __init__(self, num_labels, lstm_hidden_dim=128, lstm_layers=1, dropout=0.3):
        super(RobertaLSTMClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(input_size=self.roberta.config.hidden_size,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        pooled_output = lstm_output[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}

num_labels = len(label_encoder.classes_)
model = RobertaLSTMClassifier(num_labels=num_labels)

# ----------------------------
# METRICS
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ----------------------------
# TRAINING CONFIGURATION
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    report_to="none",
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    save_total_limit=2
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=6,
    early_stopping_threshold=0.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

# ----------------------------
# TRAIN
# ----------------------------
train_result = trainer.train()

# ----------------------------
# METRICS
# ----------------------------
train_metrics = trainer.evaluate(eval_dataset=dataset['train'])
val_metrics = trainer.evaluate(eval_dataset=dataset['validation'])

print("\n📊 Training Metrics:")
print(f"Accuracy:  {train_metrics['eval_accuracy']:.4f}")
print(f"Precision: {train_metrics['eval_precision']:.4f}")
print(f"Recall:    {train_metrics['eval_recall']:.4f}")
print(f"F1-score:  {train_metrics['eval_f1']:.4f}")

print("\n📊 Validation Metrics:")
print(f"Accuracy:  {val_metrics['eval_accuracy']:.4f}")
print(f"Precision: {val_metrics['eval_precision']:.4f}")
print(f"Recall:    {val_metrics['eval_recall']:.4f}")
print(f"F1-score:  {val_metrics['eval_f1']:.4f}")

test_preds_output = trainer.predict(dataset['test'])
test_preds = np.argmax(test_preds_output.predictions, axis=1)
test_labels_arr = np.array(test_labels)

test_acc = accuracy_score(test_labels_arr, test_preds)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels_arr, test_preds, average='weighted', zero_division=0)

print("\n📊 Test Metrics:")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-score:  {test_f1:.4f}")

print("\n🗞 Classification Report:\n")
print(classification_report(
    test_labels_arr,
    test_preds,
    labels=np.unique(test_labels_arr),
    target_names=label_encoder.inverse_transform(np.unique(test_labels_arr))
))

# ----------------------------
# SAVE MODEL & TOKENIZER
# ----------------------------
model_save_path = "./cbh_roberta_lstm_model"
tokenizer.save_pretrained(model_save_path)
torch.save(model.state_dict(), f"{model_save_path}/pytorch_model.bin")
print("\n✅ Model saved to ./cbh_roberta_lstm_model")

# ----------------------------
# TRAINING VS TESTING ACCURACY LINE PLOT
# ----------------------------
train_acc = train_metrics["eval_accuracy"]

labels = ['Training Accuracy', 'Testing Accuracy']
values = [train_acc, test_acc]

plt.figure(figsize=(6, 5))
plt.plot(labels, values, marker='o', linestyle='-', color='blue')
plt.ylim(0.8, 1.0)
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy')
for index, value in enumerate(values):
    plt.text(index, value + 0.005, f"{value:.4f}", ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("training_vs_testing_accuracy.png")
plt.show()

# ----------------------------
# TRAINING VS VALIDATION ACCURACY LINE PLOT
# ----------------------------
train_val_labels = ['Training Accuracy', 'Validation Accuracy']
train_val_values = [train_acc, val_metrics['eval_accuracy']]

plt.figure(figsize=(6, 5))
plt.plot(train_val_labels, train_val_values, marker='o', linestyle='-', color='green')
plt.ylim(0.8, 1.0)
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
for index, value in enumerate(train_val_values):
    plt.text(index, value + 0.005, f"{value:.4f}", ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("training_vs_validation_accuracy.png")
plt.show()
