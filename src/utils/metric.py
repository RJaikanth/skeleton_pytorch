from sklearn import metrics


def accuracy_fn(true, pred):
    pred = pred.argmax(dim=1)
    pred, true = pred.cpu(), true.cpu()
    return metrics.accuracy_score(y_true=true, y_pred=pred)
