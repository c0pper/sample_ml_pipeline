def clean_text(text):
    import re
    import string

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
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

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