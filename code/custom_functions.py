from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
from google.colab import files


def evaluate(true, pred):
    precision = precision_score(true, pred, average = 'weighted')
    recall = recall_score(true, pred, average = 'weighted')
    f1 = f1_score(true, pred, average = 'weighted')
    acc = accuracy_score(true, pred)
    index = 'result'
    d = {'accuracy': round(acc,4), 'precision': round(precision,4), 'recall': round(recall,4), 'f1 score': round(f1,4) }
    df = pd.DataFrame(d,index=["results"])
    sns.heatmap(pd.DataFrame(confusion_matrix(true, pred)), annot=True, cmap='Oranges', fmt='.7g');
    return df

def prediction(data, name, download = False):
    df = pd.DataFrame(data = data)
    df.index.names = ['id']
    df.rename(columns = {0:'difficulty'}, inplace = True)
    file_name = name + ".csv"
    df.to_csv(file_name)
    if download == True:
      files.download(file_name)
    return df.head()
