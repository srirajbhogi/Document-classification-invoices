import streamlit as st
import joblib
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import fitz 
import plotly.express as px
import base64

st.set_page_config(page_title="Document Classification App", page_icon="📄")

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
logistic_regression_model_tfidf = joblib.load('logistic_regression_model_tfidf.pkl')
logistic_regression_model_bert = joblib.load('logistic_regression_model_bert.pkl')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

true_labels = {
    "131-Amit-Singh.pdf": "Eyewear",
    "Invoice_2024-06-22_16-17-54_1.pdf": "FMCG",
    "1.pdf": "Footwear",
    "001.pdf": "Hardware",
    "Invoice - 2024-05-09T101347.419.pdf": "IT",
    "Invoice 00001.pdf": "Jewellery",
    "09021 _ Netmeds Pharmacy.pdf": "Pharma"
}

df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("images.jpeg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1534644107580-3a4dbd494a95?q");
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.sidebar.title("📚 Document Classification")
st.sidebar.markdown("Welcome to the Document Classification! Use the sections below to navigate.")

with st.sidebar.expander("📄 Document Classification"):
    if st.button("Go to Document Classification"):
        st.session_state['page'] = "Document Classification"

with st.sidebar.expander("ℹ️ About the Project"):
    if st.button("Go to About the Project"):
        st.session_state['page'] = "About the Project"

if 'page' not in st.session_state:
    st.session_state['page'] = "Document Classification"

if st.session_state['page'] == "Document Classification":
    st.title('Document Classification')
    st.write('Document classification is a process that involves categorizing documents into predefined categories based on their content.')
    st.write('Upload multiple PDF documents to classify their type.')

    uploaded_files = st.file_uploader("Choose documents...", type=["pdf"], accept_multiple_files=True)
    embedding_choice = st.selectbox(
        'Select embedding method',
        ('BERT','TF-IDF')
    )

    if uploaded_files and len(uploaded_files) > 0:
        st.write("Processing PDFs...")

        progress_bar = st.progress(0)
        results = []
        total_files = len(uploaded_files)

        results = []
        for i,uploaded_file in enumerate(uploaded_files):
            text = extract_text_from_pdf(uploaded_file)

            if embedding_choice == 'BERT':
                features = get_bert_embeddings(text)
                prediction = logistic_regression_model_bert.predict(features)
                confidence_score = logistic_regression_model_bert.predict_proba(features).max()
            else:
                features = tfidf_vectorizer.transform([text])
                prediction = logistic_regression_model_tfidf.predict(features)
                confidence_score = logistic_regression_model_tfidf.predict_proba(features).max()
            
            results.append({
                'File Name': uploaded_file.name,
                'Predicted Label': prediction[0],
                'Confidence Score': confidence_score
            })

            progress_bar.progress((i + 1) / total_files)

        results_df = pd.DataFrame(results)
        st.write("Classification Results")
        st.dataframe(results_df)

elif st.session_state['page'] == "About the Project":
    st.title("About the Project")
    st.write("""
    **Project Overview**
    📑 **Objective:** To develop a system that automatically classifies documents into predefined categories based on their content. This system can streamline workflows, organize documents, and enhance accessibility.
    
    📂 **Scope:**
    - **Document Types:** PDFs,images, etc.
    - **Categories:** Predefined classes such as invoices, contracts, reports, etc.
    - **Tech Stack:** Python, machine learning libraries (Scikit-Learn, TensorFlow, PyTorch), OCR tools (Tesseract, Google Cloud Vision), and possibly a web framework for deployment (Streamlit).

    **Project Phases**
    1. **Project Planning**
       - 📍 **Define Objectives:** Identify the purpose and goals of the classification system.
       - 📋 **Scope and Requirements:** Determine the types of documents, classification categories, and performance metrics.

    2. **Data Collection and Preparation**
       - 📥 **Data Acquisition:** Gather a diverse set of documents that represent each category.
       - 🏷️ **Data Labeling:** Annotate documents with their corresponding categories.
       - 🗃️ **Data Storage:** Organize the data for easy access and processing.

    3. **Data Preprocessing**
       - 📝 **Text Extraction:** Extract text from documents using OCR or other methods if necessary.
       - 🧹 **Text Cleaning:** Remove noise, normalize text, and handle any inconsistencies.
       - 🔍 **Feature Extraction:** Convert text into numerical features using techniques like TF-IDF, word embeddings, etc.

    4. **Model Development**
       - 🤖 **Select Algorithms:** Choose suitable machine learning or deep learning algorithms based on the problem (e.g., Logistic Regression, Neural Networks).
       - 🏋️ **Model Training:** Train the model on the labeled data using appropriate algorithms.
       - ⚙️ **Hyperparameter Tuning:** Optimize model parameters to improve performance.

    5. **Model Evaluation**
       - 📊 **Validation:** Use techniques like cross-validation to assess model performance.
       - 📈 **Metrics:** Evaluate using accuracy, precision, recall, F1-score, etc.
       - 🔍 **Error Analysis:** Analyze misclassifications to understand and address potential issues.

    6. **Model Deployment**
       - 🔌 **Integration:** Incorporate the trained model into a production environment.
       - 🌐 **User Interface:** Create a front-end interface for users to upload and classify documents with Streamlit.

    7. **Monitoring and Maintenance**
       - 📉 **Performance Monitoring:** Continuously monitor the system’s performance in the production environment.
       - 🔄 **Model Updating:** Periodically retrain the model with new data to keep it updated and accurate.
       - 📢 **Feedback Loop:** Collect user feedback to improve the system.
    """)

    st.image("flowchart.png", caption="Project Overview", use_column_width=True)





























