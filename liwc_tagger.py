import json
import nltk

def get_features(file):
    file_liwc = open(file, 'r')
    liwc_data_raw = file_liwc.read()

    liwc = json.loads(liwc_data_raw)

    features = []

    for k in liwc.keys():
        features.append(k)
    
    file_liwc.close()
    return features, liwc

def tag(sentence, features, liwc):
    words = nltk.word_tokenize(sentence)

    tags = []
    true_feature = []

    for word in words:
        # lower case
        word = word.lower()
        for feature in features:
            if feature in true_feature:
                continue
            all_words = liwc[feature]


            for w in all_words:
                if word.find(w) != -1:
                    true_feature.append(feature)
                    # feature confirmed, no need to confirm again
                    break
    for feature in features:
        if feature in true_feature:
            tags.append(1)
        else:
            tags.append(0)
    
    return tags

def main():
    features, liwc = get_features('liwc_feature.json')
    tags = tag('Donald Trump was elected as the 45th president of the united states of america last night', features, liwc)
    print(tags)

    splits = tags.split()
    print(len(splits))

if __name__ == '__main__':
    main()