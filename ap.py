import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Medical Urgency AI", page_icon="🏥")

st.title("🏥 Hospital Emergency Triage AI System")
st.write("24/7 AI Classifier for Patient Urgency Levels.")

@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="usman-1040/my_custom_urgency_model")

classifier = load_classifier()
labels_map = {0: "High 🚨 (Emergency)", 1: "Low 🟢 (Routine)", 2: "Medium ⚠️ (Moderate)"}

user_input = st.text_area("Patient Symptoms / Message:", placeholder="Type symptoms in English...")

if st.button("Predict Urgency", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing condition..."):
            result = classifier(user_input)
            label_id = int(result[0]['label'].split('_')[-1])
            urgency = labels_map[label_id]
            confidence = result[0]['score'] * 100
            
            st.divider()
            st.subheader(f"Urgency Level: {urgency}")
            st.write(f"**Confidence Score:** {confidence:.1f}%")
    else:
        st.warning("Please enter a patient description first.")


st.divider()
with st.expander("ℹ️ About the Model & Training Code"):
    st.markdown("""
    **Model Architecture:** Fine-tuned Transformer for Text Classification  
    **Frameworks Used:** PyTorch, Hugging Face Transformers, Datasets  
    
    📄 **View Training Code & Notebook:** [GitHub Repository](https://github.com/coversiphone84-cmyk/medical-urgency-app)  
    🤗 **View Model Weights:** [Hugging Face Model Hub](https://huggingface.co/usman-1040/my_custom_urgency_model)
    """)
