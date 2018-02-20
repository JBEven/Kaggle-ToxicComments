import pandas as pd
import re

def getDataFrom(folderPath): # Getting train.csv and test.csv (into data frame) from given folder
    if(type(folderPath) != type('truc')):
        return null, null
    train = pd.read_csv(folderPath+'/train.csv')
    train_input = train[['comment_text']]
    rating_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_output = train[rating_columns]
    test = pd.read_csv(folderPath+'/train.csv')
    return train_input, train_output, test

def getFeatures(dataset,extra,word,char, tfidfWord=False, tfidfChar=False): # extract from the dataset 'comment_text' column the features needed (given from boolean extra,word,char); tfidfWord and tfidfChar indicate if tfidf is preffered over bag of words.
    if extra:
        extraFeats = getExtraFeatures(dataset)
    else:
        extraFeats = 0
    if word:
        word_feats, wordPipe = getWordRepresentation(dataset,tfidfWord)
    else:
        word_feats = 0
    if char:
        char_feats, charPipe = getCharRepresentation(dataset,tfidfChar)
    else:
        char_feats = 0
    return extra, word_feats, char_feats, wordPipe, charPipe

def getWordRepresentation(dataset,tfidfWord): #Get either bag of word or tfidf or dataset 'comment
    pass

def getCharRepresentation(dataset,tfidfChar):
    pass

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

    extra_features_list=['nb_char','nb_word','dot','exclamation','semi-colon','comma','question','double_dot','parenthesis','hooks']
    # Cleaning special characters, lowering cases, punctuation and stopwords are dealt with sklearn
    # We are left with some lemmatization to deal with

    return extrafeats, extra_features_list
