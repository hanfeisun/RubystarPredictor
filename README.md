# RubystarPredictor
Classifiers for chatbot log

## How to run

1.  Run `pip3 install sklearn numpy pandas`
2.  Run `python3 main.py`

## How to add new features

1. For sentence-level features, add them within `sentence_embedding_func`
2. For session-level features, add them within `session_embedding_func`

## Baseline Performance

- Kappa score is 0.156174

- Accuracy is 0.672835

## Add LIWC features (5 sentences)

Logistic Regression

- Kappa score is 0.190853
- Accuracy is 0.676536

Linear SVM

- Kappa score is 0.170880
- Accuracy is 0.675796


## HAN

- Kappa score is 0.134208
- Accuracy is 0.678016

## HAN with 32 LIWC

- Kappa score is 0.250078
- Accuracy is 0.687639
