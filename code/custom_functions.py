from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
import numpy as np
from google.colab import files
from tqdm import tqdm
from nltk.corpus import stopwords        
from nltk.tokenize import word_tokenize, sent_tokenize 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import string


import spacy
from spacy import displacy
sp = spacy.load("fr_core_news_sm")


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


def pred_compare_df(X_test, y_test, y_pred):
    tempdf_1 = pd.concat([X_test, y_test], axis = 1).reset_index(drop=True)
    y_pred_df = pd.Series(y_pred)
    tempdf_2 = pd.concat([tempdf_1, y_pred_df], axis =1).rename(columns = {0: 'predicted difficulty'}) 
    tempdf_3 = pd.Series(tempdf_2['difficulty'] == tempdf_2['predicted difficulty'])
    final_df = pd.concat([tempdf_2, tempdf_3], axis = 1).rename(columns = {0: 'correct prediction'}) 
    return final_df

def compare(df, i):
  temp = df.iloc[i]
  print("sentence:\t\t%s\ndifficulty:\t\t%s\npredicted difficulty:\t%s\ncorrect prediction:\t%s" % (temp["sentence"], temp["difficulty"], temp["predicted difficulty"], temp['correct prediction']))

def prediction(data, name, download = False):
    df = pd.DataFrame(data = data)
    df.index.names = ['id']
    df.rename(columns = {0:'difficulty'}, inplace = True)
    file_name = name + ".csv"
    df.to_csv(file_name)
    if download == True:
      files.download(file_name)
    return df.head()

def spacy_tokenizer(sentence):
    doc = sp(sentence)
    stop_words = nltk.corpus.stopwords.words("french")
    punctuations = string.punctuation
  
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc ]
    mytokens = [ word for word in mytokens if word not in punctuations and word not in stop_words ]
    return mytokens

def get_info(df):
 
    text_length = []                          
    number_of_sentences = []
    number_of_words = []
    sent_length_avg = []
    words_length_avg = []
    number_of_words_after_lemma_stop = []
    longest_word_size = []
    
    for text in tqdm(df['sentence'].values):
        
      initial_length = len(text)
      text_length.append(initial_length)

      num_sentences = len(sent_tokenize(text))
      number_of_sentences.append(num_sentences)
        
      punctuations = string.punctuation
      text2 = text.lower()
      text2 = word_tokenize(text2)
      text2 = [word for word in text2 if word not in punctuations]
      num_words = len(text2)
      number_of_words.append(num_words)

      sent_length_avg.append(num_words/num_sentences)
        
      words_length_avg.append(initial_length/num_words)

      text = sp(text)
      text = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in text]
      text = [word for word in text if not word in spacy.lang.fr.stop_words.STOP_WORDS and word not in punctuations]

      num_words_after_lemma_stop = len(text)
      number_of_words_after_lemma_stop.append(num_words_after_lemma_stop)

      word_len = [len(w) for w in text2]
      longest_word_size.append(np.max(word_len))
        
    final_df = pd.concat([pd.Series(text_length), pd.Series(number_of_sentences),
                             pd.Series(number_of_words), pd.Series(sent_length_avg),
                             pd.Series(words_length_avg), pd.Series(number_of_words_after_lemma_stop),
                             pd.Series(longest_word_size)], axis = 1)
    final_df.columns = ["text_length", "number_of_sentences", "number_of_words",
                           "sent_length_avg", "words_length_avg",
                           "number_of_words_after_lemma_stop", "longest_word_size"]
    
    return final_df
