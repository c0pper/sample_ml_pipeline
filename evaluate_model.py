def print_confusion_matrix(yvalid, predicted, target_names):
    from sklearn import metrics
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = metrics.confusion_matrix(yvalid, predicted)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(xticks_rotation="vertical")


def evaluate_model(model, xvalid_tfidf, yvalid, target_names):
    from sklearn import metrics
    import numpy as np

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