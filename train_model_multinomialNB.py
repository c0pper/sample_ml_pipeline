def train_model_multinomialNB(xtrain_tfidf, ytrain):
    from sklearn.naive_bayes import MultinomialNB

    print("\n\n[+] Training model")
    clf = MultinomialNB()
    clf.fit(xtrain_tfidf, ytrain)
    return clf