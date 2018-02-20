import pandas as pd
import re

def getDataFrom(folderPath): # Getting train.csv and test.csv (into data frame) from given folder
    if(type(folderPath) != type('truc')):
        return null, null
    train = pd.read_csv(folderPath+'/train.csv')
    train_input = train[['comment_text']]
    rating_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_output = train[rating_columns]
    test = pd.read_csv(folderPath+'/test.csv')
    return train_input, train_output, test

def getFeatures(dataset,extra,word,char, wordPipe, charPipe,test=False): # extract from the dataset 'comment_text' column the features needed (given from boolean extra,word,char); tfidfWord and tfidfChar indicate if tfidf is preffered over bag of words.
    if extra:
        extra_feats  = getExtraFeatures(dataset)
    else:
        extra_feats = 0
    if word:
        if test:
            word_feats = wordPipe.transform(dataset['comment_text'])
        else:
            word_feats = wordPipe.fit_transform(dataset['comment_text'])
    else:
        word_feats = 0
    if char:
        if test:
            char_feats = charPipe.transform(dataset['comment_text'])
        else:
            char_feats = charPipe.fit_transform(dataset['comment_text'])
    else:
        char_feats = 0
    return extra_feats, word_feats, char_feats, wordPipe, charPipe

def getTestFeatures(dataset,extra,word,char, wordPipe, charPipe):
    return getFeatures(dataset,extra,word,char, wordPipe, charPipe, True)
    
def getExtraFeatures(dataset): # Getting extra features from 'comment_text' variable
    extrafeats = pd.DataFrame()
    
    # Before any feature extraction we are going to do some cleaning while keeping some usable features
    # Counting chars
    extrafeats['nb_char'] = dataset['comment_text'].map(lambda x:len(x))
    
    # nb of words
    extrafeats['nb_word'] = dataset['comment_text'].map(lambda x:len(re.findall(r'\w+', x)))

    # Counting % of UpperCase
    extrafeats['per_upper_case'] = dataset['comment_text'].map(lambda x:sum(1 for c in x if c.isupper())/len(x))

    # Counting punctuations such as . ! ; , ? ) or (
    extrafeats['dot'] = dataset['comment_text'].map(lambda x:x.count('.')/len(x))
    extrafeats['exclamation'] = dataset['comment_text'].map(lambda x:x.count('!')/len(x))
    extrafeats['semi-colon'] = dataset['comment_text'].map(lambda x:x.count(';')/len(x))
    extrafeats['comma'] = dataset['comment_text'].map(lambda x:x.count(',')/len(x))
    extrafeats['question'] = dataset['comment_text'].map(lambda x:x.count('?')/len(x))
    extrafeats['double_dot'] = dataset['comment_text'].map(lambda x:x.count(':')/len(x))
    extrafeats['parenthesis'] = dataset['comment_text'].map(lambda x:(x.count(')')+x.count('('))/len(x))
    extrafeats['hooks'] = dataset['comment_text'].map(lambda x:(x.count(']')+x.count('['))/len(x))

    return extrafeats
