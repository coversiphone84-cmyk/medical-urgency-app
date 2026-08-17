# 🏥 Hospital Emergency Triage AI System

An AI-powered web application that automatically classifies patient symptom descriptions into **Emergency Urgency Levels** (High, Medium, Low) using a fine-tuned Transformer model.

---

## 🌟 Key Features
* ⚡ **Real-time Urgency Prediction:** Instant classification into High 🚨, Medium ⚠️, or Low 🟢 urgency.
* 📊 **Confidence Score:** Provides AI decision confidence percentage for clinical assistance.
* ☁️ **Cloud Hosted:** Fine-tuned model hosted on Hugging Face Hub and web UI deployed via Streamlit.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Model Framework:** Hugging Face `transformers`, PyTorch
* **Web UI Framework:** Streamlit
* **Deployment:** Streamlit Cloud & Hugging Face Hub

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/coversiphone84-cmyk/medical-urgency-app.git](https://github.com/coversiphone84-cmyk/medical-urgency-app.git)
   cd medical-urgency-app





   the backend code used for your practice by using ML and other techniques fine tuning from hugging face

   !pip install transformers datasets torch pandas accelerate scikit-learn

   import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline

# 1. CSV Loading
df = pd.read_csv("patient_conversations.csv")

# 2. Label Encoding (0: High, 1: Low, 2: Medium)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["urgency"])

# 3. Context Combining
df["text"] = "Symptom: " + df["symptom"].fillna("") + " | Message: " + df["patient_message"].fillna("")

# 4. Dataset & Tokenizer Setup
dataset = Dataset.from_pandas(df[["text", "label"]])
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# 5. Training Settings 
training_args = TrainingArguments(
    output_dir="./my_custom_urgency_model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    logging_steps=5,
    save_strategy="epoch"
)

# 6. Trainer Setup & Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Training shuru ho rahi hai...")
trainer.train()


after training on google collab you can test from there

# 7. Saving Model
trainer.save_model("./my_custom_urgency_model")
tokenizer.save_pretrained("./my_custom_urgency_model")
print("Model fine-tune aur save ho gaya hai!")


classifier = pipeline("text-classification", model="./my_custom_urgency_model")

test_msg = input("Please elaborate your current condition: ")
result = classifier(test_msg)

# Correct Mapping (0: High, 1: Low, 2: Medium)
labels_map = {0: "High", 1: "Low", 2: "Medium"}
predicted_class = int(result[0]['label'].split('_')[-1])

print(f"\nPredicted Urgency Level : {labels_map[predicted_class]}")
print(f"Confidence Score        : {result[0]['score'] * 100:.1f}%")




   
