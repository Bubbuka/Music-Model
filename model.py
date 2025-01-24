from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm 
from test import tokenizer
import os


if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("GPU is available. Using GPU.")
else:
    device = torch.device("cpu")  # Use CPU
    print("GPU is not available. Using CPU.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load your tokenized dataset
# Define the root directory containing subfolders with MIDI files
root_dir = Path("/Users/mateoterreni/Desktop/Ai_music_project/Dataset")

# Recursively collect all MIDI file paths
midi_files = list(root_dir.glob("**/*.mid"))  # Finds all .mid files in subfolders

import random

# Define the fraction of the dataset to use (e.g., 10%)
subset_fraction = 0.01

# Randomly sample a subset of the MIDI files
subset_size = int(len(midi_files) * subset_fraction)
midi_files_subset = random.sample(midi_files, subset_size)

print(f"Using a subset of {len(midi_files_subset)} MIDI files (out of {len(midi_files)}).")

# Define the model configuration


# Split the dataset into training and validation sets
train_files, val_files = train_test_split(midi_files_subset, test_size=0.2, random_state=42)

# Create training dataset
train_dataset = DatasetMIDI(files_paths=train_files, tokenizer=tokenizer, max_seq_len=1024)

# Create evaluation dataset
eval_dataset = DatasetMIDI(files_paths=val_files, tokenizer=tokenizer, max_seq_len=1024)

# Create training and validation datasets
train_dataset = DatasetMIDI(files_paths=train_files, tokenizer=tokenizer, max_seq_len=1024)
val_dataset = DatasetMIDI(files_paths=val_files, tokenizer=tokenizer, max_seq_len=1024)
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,  # Size of your tokenizer's vocabulary
    n_positions=1024,  # Maximum sequence length
    n_ctx=1024,  # Context size
    n_layer=6,  # Number of Transformer layers
    n_head=8,  # Number of attention heads
    n_embd=512,  # Embedding dimension
)

# Initialize the model
model = GPT2LMHeadModel(config)

model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    overwrite_output_dir=True,  # Overwrite the output directory
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size per device
    logging_dir="./logs",  # Directory to store logs
    logging_steps=100,  # Log every 100 steps
    save_steps=500,  # Save a checkpoint every 500 steps
    evaluation_strategy="steps",  # Evaluate every `eval_steps`
    eval_steps=500,  # Evaluation frequency
    save_total_limit=2,  # Keep only the last 2 checkpoints
    report_to="tensorboard",  # Log to TensorBoard (optional)
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Metric to determine the best model
    greater_is_better=False,  # Lower eval_loss is better
    logging_first_step=True,  # Log the first step
    prediction_loss_only=True,  # Only compute loss during evaluation
)


# Define a data collator for padding sequences
data_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)

# Initialize the Trainer
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the training dataset
    eval_dataset=val_dataset,     # Pass the validation dataset
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Print outputs and predictions during training
 # For progress bars

# Function to evaluate the model and plot accuracy/confusion matrix
def evaluate_model(model, dataset, tokenizer, top_k_tokens=50):
    model.eval()
    all_preds = []
    all_labels = []

    # Iterate over the dataset with a progress bar
    for sample in tqdm(dataset, desc="Evaluating"):
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)  # Add batch dimension
        labels = torch.tensor(sample["input_ids"])  # Ground truth labels

        # Generate predictions
        with torch.no_grad():
            outputs = model(input_ids)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

        # Store predictions and labels
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

    # Plot confusion matrix for top-k tokens
    token_counts = np.bincount(all_labels)
    top_k_indices = np.argsort(token_counts)[-top_k_tokens:]  # Indices of top-k tokens
    top_k_labels = [tokenizer.vocab[i] for i in top_k_indices]  # Token names

    # Filter predictions and labels to include only top-k tokens
    filtered_preds = []
    filtered_labels = []
    for pred, label in zip(all_preds, all_labels):
        if label in top_k_indices:
            filtered_preds.append(pred)
            filtered_labels.append(label)

    # Create confusion matrix for top-k tokens
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_k_indices)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=top_k_labels,
        yticklabels=top_k_labels,
    )
    plt.xlabel("Predicted Tokens")
    plt.ylabel("True Tokens")
    plt.title(f"Confusion Matrix (Top-{top_k_tokens} Tokens)")
    plt.show()

# Evaluate the model
evaluate_model(model, dataset, tokenizer, top_k_tokens=50)

# Generate new music based on a new MIDI file
def generate_music_from_midi(model, tokenizer, midi_path, max_length=1024):
    # Load the new MIDI file
    midi = Score(midi_path)

    # Tokenize the MIDI file
    input_tokens = tokenizer(midi)

    # Truncate or pad the input tokens to the desired length
    input_tokens = input_tokens[:max_length]

    # Convert tokens to tensor and add batch dimension
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)

    # Generate new tokens
    model.eval()
    with torch.no_grad():
        generated_tokens = model.generate(
            input_tokens,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,  # Enable sampling for creativity
            top_k=50,  # Limit sampling to the top-k tokens
            top_p=0.95,  # Nucleus sampling
            temperature=0.7,  # Control randomness
        )

    # Decode the generated tokens back to MIDI
    generated_midi = tokenizer(generated_tokens.squeeze().tolist())
    generated_midi.dump("/Users/mateoterreni/Desktop/Ai_music_project/generated")
    print("Generated MIDI saved to /Users/mateoterreni/Desktop/Ai_music_project/generated")
# Generate music from a new MIDI file
generate_music_from_midi(model, tokenizer, "/Users/mateoterreni/Desktop/Ai_music_project/generated")