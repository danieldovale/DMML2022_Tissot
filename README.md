# Data Mining and Machine Learning HEC 2022
## Detecting the difficulty level of French texts


## Team TISSOT: Group members
* Giacomo Rattazzi
* João Daniel Do Vale Anes 

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Tissot_Logo.svg/2560px-Tissot_Logo.svg.png)


# Context 

The project goal was to predict the difficulty (A1 to C2) of a French text. The idea was to build models trained on **training_data.csv**. This csv contains two columns, the first being a French text and the second one being the attributed difficulty. Here is an extract:

| sentence  | difficulty |
| ------------- | ------------- |
| Je reste avec toi |  A1 |
| Un revenu devient donc une nécessité pour que l'Homme puisse accéder à la satisfaction d'avoir comblé ses désirs |  C2 |
| Tu mangeas les petits fruits dès que tu les eus cueillis | C2 | 
| Ben, on est tous débordés quoi. |  A1 | 
| Un petit garçon : Ben trois dans un nid et dans l'autre, y'en a deux | A2 | 
| J'ai été en forme toute la matinée | A2  | 

Once a model gave satisfactory accuracy, we measured its performance by testing it on unseen data. **unlabelled_test_date**
# Repository structure 
```

│
├── Code
│   └──4models.ipynb
│   └──Word2VecLR.ipynb
│   └──W2VEnsembles.ipynb
│   └──Camembert.ipynb
│   └──functions.py
│
├── Data
│   └──sample_submission.csv
│   └──training_data.csv
│   └──unlabelled_test_data.csv 
│ 
├── Documents
│   └── CustomFunctionsDocumentation.ipynb
│ 
│── README.md
```

* Code
  1. 4Models.ipynb: first part of the project. This notebook contains a short visualization of the data, and a demonstration of four different models without doing any type of data cleaning. The four models used were Logistic Regression, KNeighbors Classifier, Decision Tree Classifier, and Random Forest CLassifier. The four model employed a TF-IDF vectorizer to transform the sentences into vectors. 
  2. Word2VecLR.ipynb: Attempt at improving accuracy by using Word2Vec instead of TF-IDF for vectorization. This notebook also contains steps to retrieve additionnal information about the data, which is then used in conjunction with the Word2Vec vectors to predict text difficulty using a Logistic Regression. This new model seemingly had issues with overfitting,
  3. EnsembleW2V.ipynb: In this notebook we attempted to solve the overfitting issue that the previous Logistic Regression had using Ensemble Classifiers. We used both TF-IDF and Word2Vec vectorization.
  4. functions.py: python file containing custom functions we wrote while building our models.

* Data
  1. sample_submission.csv: sample csv file with the format needed for the kaggle submissions.
  2. training_data.csv: csv file containing 4800 rows with french sentences and an attributed difficulty.
  3. unlabelled_test_data.csv: csv file containing 1200 rows of unlabeled french sentences.

* Documents
  1. CustomFunctionsDocumentation.ipynb: Notebook documenting the custom functions we wrote in the functions.py file.



# RoadMap

![roadmap](https://user-images.githubusercontent.com/114418718/209168203-15e7a47d-48f5-46aa-819d-a3236e90f14f.png)

More information about the steps in our project is available ![here](https://)

# Results
| | Logistic Regression  | kNN | Decision Tree | Random Forest | Ensemble & W2V |
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| **Precision**         | 0.4556 | 0.4192 | 0.3158 | 0.4228 | 0.5135
| **Recall**            | 0.4667 | 0.3594 | 0.3156 | 0.4135 | 0.5169
| **F1-score**          | 0.4640 | 0.3501 | 0.3008 | 0.4000 | 0.5135
| **Accuracy**          | 0.4667 | 0.3594 | 0.3156 | 0.4135 | 0.5121
| **Submission score**  | **0.46583** | **0.34083** | **0.31833** | **0.39500** | **0.49166**



# Youtube video




