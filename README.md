# Data Mining and Machine Learning HEC 2022
## Detecting the difficulty level of French texts


## Team TISSOT: Group members
* d
* g

# Repository structure 
```
├── README.md
│
├── Code
│   └──4models.ipynb
│   └──tests.ipynb
│   └──Solution.ipynb
│
├── Data
│   └──sample_submission.csv
│   └──training_data.csv
│   └──unlabelled_test_data.csv 
│ 
```

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

Once a model gave satisfactory accuracy, we trained the model once again without splitting it to predict the difficulty of the text in **unlabelled_test_data.csv**, which contains only one column with text. 





# Results
| | Logistic Regression  | kNN | Decision Tree | Random Forest | Other |
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| Precision | 0.4556 | 0.4192 | | |
| Recall    | 0.4667 | 0.3594 | | |
| F1-score  | 0.4640 | 0.3501 | | 
| Accuracy  | 0.4667 | 0.3594 | | |

## Confusion matrixes 







