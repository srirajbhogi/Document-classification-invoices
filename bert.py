import os
import fitz
import pytesseract
from PIL import Image
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

pdf_folder_paths = ['Eyewear','FMCG','Footwear','Hardware','IT','Jewellery','Pharma','others']

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\hp\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text

def process_pdfs(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = pdf_to_text(pdf_path)
            data.append({'Filename': filename, 'Extracted Text': text, 'Folder': folder_path})
            print(f"OCR completed for {filename} in folder {folder_path}")
    return data

def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

if __name__ == "__main__":
    all_extracted_data = []
    for folder_path in pdf_folder_paths:
        extracted_data = process_pdfs(folder_path)
        all_extracted_data.extend(extracted_data)
    
    df = pd.DataFrame(all_extracted_data)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    bert_embeddings = df['Extracted Text'].apply(get_bert_embeddings)
    bert_embeddings_2d = np.array(bert_embeddings.tolist()).reshape(len(df), -1)

    bert_embeddings_df = pd.DataFrame(bert_embeddings_2d, index=df.index)
    df_final = pd.concat([df, bert_embeddings_df], axis=1)

    X = bert_embeddings_2d
    y = df['Folder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_bert = LogisticRegression(max_iter=1000)
    model_bert.fit(X_train, y_train)

    y_pred = model_bert.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Classification Report:")
    print(report)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model_bert, 'logistic_regression_model_bert.pkl')

    print("Logistic Regression model trained and saved successfully.")


