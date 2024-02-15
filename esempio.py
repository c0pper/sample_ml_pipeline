import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import re
import string
from sklearn.svm import SVC

def load_dataset(dataset_path, x_label, y_label):
    print("\n\n[+] Loading dataset\n\n")
    dataset = pd.read_csv(dataset_path, usecols=[x_label, y_label])
    return dataset


def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text) 
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)  
    return text


def preprocess_dataset(dataset, x_label, y_label):
    print("\n\n[+] Preprocessing dataset")
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(dataset[y_label])
    x = dataset[x_label]
    
    # cleaning text
    print("\n\n** Cleaning text")
    x = x.apply(clean_text)

    xtrain, xvalid, ytrain, yvalid = train_test_split(
        x, y, random_state=42, test_size=0.1, shuffle=True
    )


    tfidf_vectorizer = TfidfVectorizer()
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xvalid_tfidf = tfidf_vectorizer.transform(xvalid)

    target_names = set(lbl_enc.inverse_transform(yvalid))

    print(f"\n** Train records: {len(xtrain)}")
    listytrain = list(ytrain)
    for i in set(listytrain):
        print(f"Label: {i} (Instances: {listytrain.count(i)})")

    print(f"\n** Validation records: {len(xvalid)}")
    listyvalid = list(yvalid)
    for i in set(listyvalid):
        print(f"Label: {i} (Instances: {listyvalid.count(i)})")

    return xtrain_tfidf, xvalid_tfidf, ytrain, yvalid, target_names

def print_confusion_matrix(yvalid, predicted, target_names):
    cm = metrics.confusion_matrix(yvalid, predicted)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(xticks_rotation="vertical")

def train_model(xtrain_tfidf, ytrain):
    print("\n\n[+] Training model")
    clf = MultinomialNB()
    clf.fit(xtrain_tfidf, ytrain)
    return clf

def train_model_svm(xtrain_tfidf, ytrain):
    print("\n\n[+] Training model")
    clf = SVC()
    clf.fit(xtrain_tfidf, ytrain)
    return clf

def evaluate_model(model, xvalid_tfidf, yvalid, target_names):
    print("\n\n[+] Evaluating model")
    predicted = model.predict(xvalid_tfidf)
    accuracy = np.mean(predicted == yvalid)

    mnb_report_dict = metrics.classification_report(
        yvalid, predicted, target_names=target_names, output_dict=True
    )
    mnb_report_text = metrics.classification_report(
        yvalid, predicted, target_names=target_names
    )

    print(accuracy)
    print(mnb_report_text)
    print_confusion_matrix(yvalid, predicted, target_names)

if __name__ == "__main__":
    dataset_path = "df_gender2_4k.csv"
    dataset = load_dataset(dataset_path, x_label="texts", y_label="gender")
    xtrain_tfidf, xvalid_tfidf, ytrain, yvalid, target_names = preprocess_dataset(dataset, x_label="texts", y_label="gender")
    model = train_model_svm(xtrain_tfidf, ytrain)
    evaluate_model(model, xvalid_tfidf, yvalid, target_names)
