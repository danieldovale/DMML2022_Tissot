# Context
The project goal was to predict the difficulty (A1 to C2) of a French text. To achieve this goal, we utilized a training dataset containing 4800 labeled sentences and a separate test dataset of 1200 unlabeled sentences. Our approach involved building and training models using the labeled data from the training dataset and using these models to make predictions on the difficulty level of the sentences in the test dataset. These predictions were then submitted to a competition on Kaggle for evaluation.

# Notebook 1: 4Models.ipynb 

In this notebook we first took a look a the data to ensure that there were no irregularities, such as null values or duplicated sentences, and found that the distribution of difficulty classes was fairly even. 

We then trained four different models on the data: Simple Logistic Regression, KNeighbors Classifier, Decision Tree Classifier, and Random Forest Classifier. For the KNeighbors and Random Forest classifiers we also tuned the hyperparameters.  Before training the models, we transformed the sentences into numerical vectors using TF-IDF vectorization, which weights the importance of words based on their frequency and rarity. 

After evaluating the performance of each model on the training data, we found that the Logistic Regression model had the highest accuracy. The Decision Tree Classifier, on the other hand, performed poorly. We also observed that some of the models were biased towards certain classes, such as the KNeighbors Classifier being biased towards the A1 difficulty class. 

We then fitted the models on the full dataset, without splitting it, and used them to predict the difficulty of the unlabeled test set. We submitted the predictions to Kaggle and found that the Logistic Regression model had the highest submission accuracy. Interestingly, the Decision Tree Classifier slightly improved in accuracy compared to the other models, but it remained the lowest performing model. 

After these results, we attempted to use other models (like LinearSVC or Naive Bayes) and different tokenization techniques, such as lemmatization and removal of stop words and punctuation. Unfortunately, we were unable to achieve a submission score higher than 47% so we did not include those models in the notebook.  

# Notebook 2: Word2VecLR.ipynb

We then decided to try using word2vec instead of TF-IDF. Word2vec uses a neural network to predict the probability of a word occurring based on the context of the surrounding words, and it returns word embeddings that capture the meaning of the words. But before applying Word2Vec, we wrote a function to retrieve additional information about the sentences.

Word2Vec steps: We first tokenized all sentences in the training and test datasets, fed the vocabulary list into the word2vec model, and used the resulting vectors and additional information gathered from the custom function to train a new model. The model we used was a simple Logistic Regression, and it achieved an accuracy of 49.9%. 

To try and improve the accuracy further, we trained the model multiple times with different word2vec parameters and were able to reach 52%. However, when we applied the model to the unlabeled data, the submission score was only 46%, suggesting that the model may be overfitting to the test set and not generalizing well to unseen data. As a potential solution, we considered using ensemble classifiers, which combines the predictions of multiple individual models to make a final prediction. Ensemble models can often improve the performance of individual models and reduce the risk of overfitting.

# Notebook 3: Ensemble.ipynb 



# Notebook 4: Camembert.ipynb

⚠️ Disclaimer: This Colab notebook contains the steps to train a CamemBERT model using code that was taken from this [article.](https://www.kaggle.com/code/houssemayed/camembert-for-french-tweets-classification/notebook#CamemBERT-for-text-classification-task)

Please note that we did not write this code ourselves and were simply using it for curiosity and testing purposes. Any submissions to Kaggle using this model were done after the competition deadline.

# Conclusion

In conclusion, it would have been more effective to carefully analyze the data and thoroughly review the documentation for different models before attempting to use them for our project. Our mistake was just throwing different models with various tokenization and vectorization techniques without fully understanding their implications. This didn’t work out too well since our best model only showed a slight improvement of 3% compared to our initial logistic regression model. 

To improve the success of future projects, it would be important to take the time to thoroughly evaluate and understand the data and available models before proceeding.
