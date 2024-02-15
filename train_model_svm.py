def train_model_svm(xtrain_tfidf, ytrain):
    from sklearn.svm import SVC

    print("\n\n[+] Training model")
    clf = SVC()
    clf.fit(xtrain_tfidf, ytrain)
    return clf