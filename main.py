import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
BASE_DIR = ''

MAX_SENTENCE_PER_SESSION = 5

VALIDATION_SPLIT = 0.1
PRELOAD = False

REGRESSION = False

SENTENCE_EMBEDDING_SIZE = None
SESSION_EMBEDDING_SIZE = None

from liwc_tagger import get_features, fast_tag
features, liwc = get_features('liwc_feature.json')


def split_train_test_set(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return train, test


langdetect_count = 0

df_reviews = pd.read_csv('oneperline.csv')  # , encoding='utf-8')

df_reviews['len'] = df_reviews.text.str.len()
df_reviews['rating'] = df_reviews['rating'].round()

df_reviews = df_reviews[df_reviews['len'].between(10, 4000)]
df_reviews = df_reviews[df_reviews.rating != 3]
df_reviews['rating'] = np.where(df_reviews.rating > 3, 1, 0)

df_rev_balanced = df_reviews


def truncate_or_pad(sentence, n):
    sentence_seq = list(map(lambda x: x.strip(), sentence.split("\n")))
    L = len(sentence_seq)
    if L >= n:
        return sentence_seq[:n]
    else:
        return [""] * (n - L) + sentence_seq


pad_sentence = df_rev_balanced.text.map(lambda x: truncate_or_pad(x, MAX_SENTENCE_PER_SESSION)).values


def sentence_embedding_func(last_str, str):
    global SENTENCE_EMBEDDING_SIZE
    words = len(str.split(" "))

    if str.startswith("AAAAA"):
        embed_speaker = [0, 1]
    elif str.startswith("UUUUU"):
        embed_speaker = [1, 0]
    else:
        embed_speaker = [0, 0]

    if str.startswith("AAAAA") or str.startswith("UUUUU"):
        str = str[8:]

    embed_sentence_length = [min(len(str), 400), ]
    embed_word_length = [min(len(str.strip().split(" ")), 100), ]

    pos_word = ["love", "friend"]
    neg_word = ["stupid", "idiot", "fuck"]
    pos = False
    neg = False
    for p in pos_word:
        if p in str:
            pos = True
            break

    for n in neg_word:
        if n in str:
            neg = True
            break

    embed_sentiment = [1 if pos else 0, 1 if neg else 0]

    if len(last_str) == 0 or len(str) == 0:
        embed_overlap = [0,
                         0,
                         0,
                         0,
                         0,
                         0]
        embed_uniq_rate = [0, 0]
    else:
        this_vocab = set(str.strip("\n").split(" "))
        last_vocab = set(str.strip("\n").split(" "))
        embed_overlap = [1,
                         len(this_vocab),
                         len(last_vocab),
                         len(this_vocab | last_vocab),
                         len(this_vocab & last_vocab),
                         len(this_vocab | last_vocab) / len(this_vocab & last_vocab),
                         ]

        embed_uniq_rate = [1, len(this_vocab) / words]

    if "wh" in str or "how" in str:
        embed_question = [1]
    else:
        embed_question = [0]

    ret = embed_speaker + embed_sentence_length + embed_word_length + embed_sentiment + embed_uniq_rate + embed_question + embed_overlap
    SENTENCE_EMBEDDING_SIZE = len(ret)

    liwc_features = fast_tag(str, features=features, liwc=liwc)
    return ret + liwc_features
    # return ret


def session_embedding_func(str):
    global SESSION_EMBEDDING_SIZE

    words = len(str.replace("\n", " ").split(" "))
    turns = len(str.split("\n"))
    embed_session_length = [min(len(str), 100), min(len(str), 1000), min(len(str), 5000)]
    embed_session_words = [min(words, 100), min(words, 1000), min(words, 5000)]
    embed_session_turns = [turns]
    pos_word = ["love", "friend"]
    neg_word = ["stupid", "idiot", "fuck"]
    pos = False
    neg = False
    for p in pos_word:
        if p in str:
            pos = True
            break

    for n in neg_word:
        if n in str:
            neg = True
            break

    embed_sentiment = [1 if pos else 0, 1 if neg else 0]

    embed_uniq_rate = [len(set(str.replace("\n", " ").split(" "))) / words]

    ret = embed_session_length + embed_session_words + embed_session_turns + embed_sentiment + embed_uniq_rate
    SESSION_EMBEDDING_SIZE = len(ret)
    return ret


X_sentence_aux_embedding = np.array(
    [[sentence_embedding_func(last_sentence, sentence) for last_sentence, sentence in
      zip([""] + session[:-1:], session[::])] for session in pad_sentence], dtype="float32")
X_sentence_aux_embedding /= np.max(X_sentence_aux_embedding, axis=(0, 1,))

X_session_aux_embedding = np.array([session_embedding_func(session) for session in df_rev_balanced.text.values],
                                   dtype="float32")
X_session_aux_embedding /= np.max(X_session_aux_embedding, axis=(0,))

SIZE = X_sentence_aux_embedding.shape[0]
X_sentence_aux_embedding = X_sentence_aux_embedding.reshape(SIZE, -1)


X = np.concatenate([X_sentence_aux_embedding, X_session_aux_embedding], axis=1)
X = np.nan_to_num(X)

Y = df_rev_balanced.rating.values.astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=VALIDATION_SPLIT,
                                                    random_state=9)

from sklearn import linear_model


logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import cohen_kappa_score, accuracy_score
print("kappa score is %f" % cohen_kappa_score(y_pred, y_test))
print("accuracy is %f" % accuracy_score(y_pred, y_test))