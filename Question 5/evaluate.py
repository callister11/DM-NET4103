from sklearn.metrics import accuracy_score, mean_absolute_error

def evaluate(true_labels, predicted_labels, mask):
    true = true_labels[~mask]
    pred = predicted_labels[~mask]
    acc = accuracy_score(true, pred)
    mae = mean_absolute_error(true, pred)
    return acc, mae
