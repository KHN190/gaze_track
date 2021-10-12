from sklearn.metrics import precision_score, accur recall_score, f1_score

def evaluate(y_true, y_pred):
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"precision:\t{pre},\nrecall:\t\t{rec},\nf1:\t\t{f1}")
