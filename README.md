# 🏥 Hospital Emergency Triage AI System

An end-to-end Natural Language Processing (NLP) project that automatically classifies patient symptom descriptions into **Emergency Urgency Levels** (High 🚨, Medium ⚠️, Low 🟢) using a fine-tuned Transformer model.

---

## 🔗 Project Links & Assets
* 🌐 **Live Web App:** [Click Here to Test Demo](https://coversiphone84-cmyk-medical-urgency-app.streamlit.app/)
* 📓 **Model Training Code (Notebook):** [View Training & Fine-Tuning Notebook](./Medical_Urgency_FineTuning.ipynb)
* 🤗 **Hugging Face Model Hub:** [usman-1040/my_custom_urgency_model](https://huggingface.co/usman-1040/my_custom_urgency_model)

---

## 🌟 Key Features
* ⚡ **Real-Time Triage:** Automatically flags critical symptoms for medical emergency prioritization.
* 🎯 **Confidence Score:** Calculates prediction certainty percentage alongside emergency levels.
* ☁️ **Cloud Deployment:** Model weights hosted on Hugging Face Hub; frontend application served via Streamlit Cloud.

---

## 🧠 Model Training Pipeline
The core ML work involves fine-tuning a pretrained Transformer model for sequence classification:
* **Frameworks Used:** PyTorch, Hugging Face `transformers`, `datasets`
* **Preprocessing:** Custom text normalization and tokenization.
* **Fine-Tuning:** Trained using Hugging Face `Trainer` API with dynamic evaluation loss and accuracy metrics.
* **Full Code:** Open the repository's `.ipynb` notebook file to review the dataset preprocessing, training parameters, loss curves, and model saving pipeline.

---

## 🛠️ Tech Stack
* **Language:** Python
* **ML / Deep Learning:** PyTorch, Transformers, Hugging Face Hub
* **Frontend / UI:** Streamlit
* **Hosting:** Streamlit Cloud & Hugging Face Spaces/Hub

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/coversiphone84-cmyk/medical-urgency-app.git](https://github.com/coversiphone84-cmyk/medical-urgency-app.git)
   cd medical-urgency-app
