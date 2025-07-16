# Phishing URL Detection with Knowledge Distillation: Teacher (BERT) and Student (DistilBERT) Models

This repository contains two models for phishing URL detection: a larger, more accurate "teacher" model and a smaller, faster "student" model created using knowledge distillation.

## Model Overview

### Teacher Model: `Vyshnev/bert-phishing-classifier_teacher`

- **Architecture:** bert-base-uncased
- **Parameters:** 109 Million
- **Description:** This model is a fine-tuned version of bert-base-uncased for sequence classification on a phishing dataset. It serves as the "teacher" in the knowledge distillation process, providing the knowledge to the smaller student model.

### Student Model: `Vyshnev/bert-phishing-classifier_student`

- **Architecture:** distilbert-base-uncased (with a modified configuration of 4 layers and 8 attention heads)
- **Parameters:** 52 Million
- **Description:** This is a distilled version of the teacher model. It's significantly smaller and faster, making it more suitable for resource-constrained environments, while still maintaining a high level of performance for phishing detection.

## Training Process

The models were trained on the Vyshnev/phishing-data-classification dataset, which contains URLs labeled as either "Safe" or "Not Safe".

### Teacher Model Training
The `bert-base-uncased model` was fine-tuned for the phishing detection task. During this process, the majority of the base model's parameters were frozen to leverage the pre-trained knowledge, while the pooling and classification layers were unfrozen and trained on the specific dataset. This approach helps to prevent overfitting and adapt the model effectively to the task.

### Knowledge Distillation (Student Model Training)
The student model, a modified distilbert-base-uncased, was trained using knowledge distillation. This technique involves training the smaller student model to mimic the behavior of the larger teacher model. The training process uses a combined loss function:

- **Distillation Loss (KL Divergence):** This encourages the student model's output probabilities to match the "soft labels" (the full probability distribution) of the teacher model. This helps the student learn the nuances of the teacher's predictions.
- **Hard-Label Loss (Cross-Entropy):** The standard cross-entropy loss is also used with the ground-truth labels to ensure the student model learns the correct classifications.
The final loss is a weighted average of these two losses, controlled by a hyperparameter `alpha.`

### Hyperparameters
| **Hyperparameters** | **Teacher Model Value** | **Student Model Value** |
| :--- | :--- | :--- | 
| Learning Rate | 2e-4 | 1e-4 |
| Batch Size | 8 | 32 |
| Number of Epochs | 10 | 5 |
| Batch Size | - | 2.0 |
| Learning Rate | - | 0.5 |

## Evaluation Results
### Test Set Performance
| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher** | 0.8778 | 0.8718 | 0.8908 | 0.8812 |
| **Student** | 0.9022 | 0.9384 | 0.8646 | 0.9000 |

### Validation Set Performance
| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher** | 0.8844 | 0.8811 | 0.8889 | 0.8850 |
| **Student** | 0.9156 | 0.9561 | 0.8711 | 0.9116 |

As you can see, the student model not only is significantly smaller but also achieves comparable and, in some metrics, even better performance than the teacher model on the test and validation sets.

## How to Use the Models
You can use these models for inference with the transformers library in Python.

1. Install the necessary libraries:
```
pip install torch transformers
```

2. Load the desired model and its corresponding tokenizer:
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the student model from the Hub
student_model_name = "Vyshnev/bert-phishing-classifier_student"  # for teacher model "Vyshnev/bert-phishing-classifier_teacher"
model = AutoModelForSequenceClassification.from_pretrained(student_model_name)

# The student model uses the DistilBERT tokenizer 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # for teacher tokenizer bert-base-uncased

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```
3. Perform Inference:
```
input_text = "google.com"  # Replace with the URL you want to classify
inputs = tokenizer(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Map prediction to label using the model's configuration
predicted_label = model.config.id2label[predictions.item()]
print(f"The URL '{input_text}' is predicted as: {predicted_label}")
```