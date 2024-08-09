import os
import fitz 
import pytesseract
from PIL import Image
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords')

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\hp\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

pdf_folder_paths = ['Eyewear','FMCG','Footwear','Hardware','IT','Jewellery','Pharma','others']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    cleaned_text = ' '.join(words)
    return cleaned_text

def pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def process_pdfs(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = pdf_to_text(pdf_path)
            data.append({'Filename': filename, 'Extracted Text': text, 'Folder': folder_path})
            print(f"OCR completed for {filename} in folder {folder_path}")
    return data
 
if __name__ == "__main__":
    all_extracted_data = []
    for folder_path in pdf_folder_paths:
        extracted_data = process_pdfs(folder_path)
        all_extracted_data.extend(extracted_data)
   
    df = pd.DataFrame(all_extracted_data)
   
    df['Cleaned Text'] = df['Extracted Text'].apply(clean_text)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(df['Cleaned Text'])

    df['Label'] = df['Folder']

    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['Label'], test_size=0.2, random_state=42)

    model_tfidf = LogisticRegression(penalty='l2', C=0.2)
    model_tfidf.fit(X_train, y_train)

    y_pred = model_tfidf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(model_tfidf, 'logistic_regression_model_tfidf.pkl')
 
    print("Model and vectorizer saved successfully.")








