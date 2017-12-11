# RubystarPredictor
Classifiers for chatbot log

## How to run

1.  Run `pip3 install sklearn numpy pandas`
2.  Run `python3 main.py`

## How to add new features

1. For sentence-level features, add them within `sentence_embedding_func`
2. For session-level features, add them within `session_embedding_func`

## Baseline Performance

kappa score is 0.156174
accuracy is 0.672835

