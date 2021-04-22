import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
import pandas as pd
import numpy as np
import os
from transformers import AutoModel, BertTokenizerFast, BertTokenizer
from transformers import RobertaTokenizer
# For preprocessing ASCII...
from unidecode import unidecode
import demoji
import re
import copy
import string
from itertools import groupby
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
demoji.download_codes()

# specify GPU
# device = torch.device("cuda")

def process_data_test(data_path):
    data = pd.read_csv(data_path, sep='\t')
    data["text"] = data["text"].apply(lambda x:unidecode(x))  
    data["text"] = data["text"].apply(lambda x:re.sub(r'http\S+', "URL", x))
    sentences = data["text"]
    return np.array(sentences)

def preprocess_cleaning(df):
    '''
    Convert non-ascii to ascii
    Count URL, emoji, punc, hashtag, mentions
    convert emoji to text
    convert hashtag using camel case
    lower text
    '''

    stop_words = stopwords.words('english')
    EMOJI_TO_TEXT = demoji.findall((' ').join(df['tweet_text'].to_list()))
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

    def lemmatize_words(text):
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

    def clean_text(text):
        '''Make text lowercase, remove text in square brackets, remove links, remove user mention,
        remove punctuation, remove numbers and remove words containing numbers.'''
        
        text = re.sub('(#[A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text))  # Split by camel case
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('@\w+', '', text) # mentions
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punc
        text = re.sub('\n', '', text)
        text = re.sub(r'(.)\1+', r'\1\1', text) # char repeated more than twice. ex hellllp -> hellp
        
        return text

    def emoji_to_text(text):
        return ' '.join([EMOJI_TO_TEXT.get(i, i) for i in text.split(' ')])

    df['num_url']=df['tweet_text'].apply(lambda x:x.count('URL'))
    df['num_user_id']=df['tweet_text'].apply(lambda x:x.count('USERID'))
    df['num_emoji'] = df['tweet_text'].apply(lambda x:len([i for i in x if i in EMOJI_TO_TEXT]))
    
    df['tweet_text']=df['tweet_text'].apply(lambda x:emoji_to_text(x))
    df['tweet_text']=df['tweet_text'].apply(lambda x:unidecode(x))
    df['tweet_text']=df['tweet_text'].apply(lambda x:lemmatize_words(x))
    
    df['has_url']=(df['num_url']>0).astype(int)
    df['has_emoji']=(df['num_emoji']>0).astype(int)
    df['num_hashtags'] = df['tweet_text'].str.findall(r'#(\w+)').apply(lambda x : len(x))
    df['num_user_mention'] = df['tweet_text'].str.findall(r'@(\w+)').apply(lambda x : len(x))
    df['num_punctuation'] = df['tweet_text'].str.replace(r'[\w\s#]+', '').apply(lambda x : len(x))
    

    df['text_cleaned'] = df['tweet_text'].apply(clean_text)
    # Remove stop words
    df['text_cleaned'] = df['text_cleaned'].str.split().apply(lambda x: [word for word in x if word not in stop_words]).apply(lambda x: ' '.join(x))
    
    return df

def convert_label(labels):
    for i in range(labels.shape[1]):
        # Hardcode encoding : 0:no,1:yes,2:nan
        if i in [1, 2, 3, 4]:
            # Include nan
            col = np.copy(labels[:,i])
            col[np.where(col == 'no')] = 0
            col[np.where(col == 'yes')] = 1
            col[np.where(col == 'nan')] = 2
            labels[:, i] = np.copy(col).astype(np.int32)
        else:
            col = np.copy(labels[:,i])
            col[np.where(col == 'no')] = 0
            col[np.where(col == 'yes')] = 1
            labels[:, i] = np.copy(col).astype(np.int32)
    return labels

def summarise_data(data_path='data/english/v3/v3/covid19_disinfo_binary_english_train.tsv'):
    data = pd.read_csv(data_path, sep='\t')
    data=data.dropna(subset=['q7_label', 'q6_label'])
    tem = data.iloc[:, 2:].fillna('nan')
    print('Training Label Statistics')
    print('--------')
    print(tem.describe())
    print('--------')
    for i in range(7):
        print(i); print(tem['q'+str(i+1)+'_label'].value_counts()); print('-----');

def generate_class_weights(data_path='data/english/v1/v1/covid19_disinfo_binary_english_train.tsv', return_weights=False):
    print("Generating Class Weights")
    if not os.path.exists('data/class_weights'):
        os.mkdir('data/class_weights')

    data = pd.read_csv(data_path, sep='\t')
    data=data.dropna(subset=['q7_label', 'q6_label'])
    tem = data.iloc[:, 2:].fillna('nan')
    WTS = []
    for i in range(7):
        x = np.array(tem['q' + str(i+1) + '_label'])
        wts = []
        y_count = len(np.where(x == 'yes')[0])
        n_count = len(np.where(x == 'no')[0])
        n_s = y_count + n_count
        wts.append(n_s / (2.0 * n_count))
        wts.append(n_s / (2.0 * y_count))
        # class_count = len(np.where(x == 'no')[0])
        # wts.append(1.0 / (class_count / (1.0 * len(x)) + 1.0))
        # class_count = len(np.where(x == 'yes')[0])
        # wts.append(1.0 / (class_count / (1.0 * len(x)) + 1.0))

        if i in [1, 2, 3, 4]:
            class_count = len(np.where(x == 'nan')[0])
            # wts.append(class_count / (1.0 * len(x)))
            # wts.append(np.min([wts[0], wts[1]])/10.0) # Small weight for nans
            wts.append(np.min([wts[0], wts[1]])/15.0)

        wts = np.array(wts)
        # wts = wts / np.sum(wts)
        WTS.append(wts)
        np.save(os.path.join('data/class_weights', 'q' + str(i+1) + '.npy'), wts)
    
    if return_weights:
        return WTS

'''
data_path : path in string where data is held

Returns :
	- sentences : numpy array of sentences
	- labels : categorical numeric values of each class of shape (num_sentences, 7) for each question
	- les : LabelEncoder class objects for decoding the classes later on
'''
def process_data(data_path):
    data = pd.read_csv(data_path, sep='\t')
    print("Dropping rows that contain nan in Q6_label or Q7_label")
    data=data.dropna(subset=['q7_label', 'q6_label'])  
    # TODO : Check removal for not English data
    data["tweet_text"] = data["tweet_text"].apply(lambda x:unidecode(x))  
    sentences = data["tweet_text"]
    labels = np.array(data.iloc[:, 2:].fillna('nan'))

    from sklearn import preprocessing
    les = []
    print("---Preprocessing---")
    for i in range(labels.shape[1]):
        # le = preprocessing.LabelEncoder()
        # le.fit(labels[:, i])

        # print("{} : {}".format(i, le.classes_))

        # labels[:, i] = le.transform(labels[:, i])
        # les.append(le)

        # Hardcode encoding : 0:no,1:yes,2:nan
        if i in [1, 2, 3, 4]:
            # Include nan
            col = np.copy(labels[:,i])
            col[np.where(col == 'no')] = 0
            col[np.where(col == 'yes')] = 1
            col[np.where(col == 'nan')] = 2
            labels[:, i] = np.copy(col).astype(np.int32)
        else:
            col = np.copy(labels[:,i])
            col[np.where(col == 'no')] = 0
            col[np.where(col == 'yes')] = 1
            labels[:, i] = np.copy(col).astype(np.int32)

        
    return np.array(sentences), labels, les

def process_bulgarian_data(data_path):
    data = pd.read_csv(data_path, sep='\t')
    data = data.fillna('nan')
    print("Dropping rows that contain nan in Q6_label or Q7_label")
    data=data.dropna(subset=['q7_label', 'q6_label'])  
    # TODO : Check removal for not English data
    data["tweet_text"] = data["tweet_text"].apply(lambda x:unidecode(x))  
    sentences = data["tweet_text"]
    labels = np.array(data.iloc[:, 2:].fillna('nan'))

    from sklearn import preprocessing
    les = []
    print("---Preprocessing---")
    for i in range(labels.shape[1]):
        # le = preprocessing.LabelEncoder()
        # le.fit(labels[:, i])

        # print("{} : {}".format(i, le.classes_))

        # labels[:, i] = le.transform(labels[:, i])
        # les.append(le)

        # Hardcode encoding : 0:no,1:yes,2:nan
        if i in [1, 2, 3, 4]:
            # Include nan
            col = np.copy(labels[:,i])
            col[np.where(col == 'no')] = 0
            col[np.where(col == 'yes')] = 1
            col[np.where(col == 'nan')] = 2
            labels[:, i] = np.copy(col).astype(np.int32)
        else:
            col = np.copy(labels[:,i])
            col[np.where(col == 'no')] = 0
            col[np.where(col == 'yes')] = 1
            labels[:, i] = np.copy(col).astype(np.int32)

        
    return np.array(sentences), labels, les

def process_test_data(data_path):
    data = pd.read_csv(data_path, sep='\t')
    print("Dropping rows that contain nan in Q6_label or Q7_label")
    data=data.dropna(subset=['q7_label', 'q6_label'])  
    data["tweet_text"] = data["tweet_text"].apply(lambda x:unidecode(x))  
    sentences = data["tweet_text"]
        
    return np.array(sentences)

def tokenize(sentences, use_type_tokens = True, padding = True, max_len = 25, bert_base='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(bert_base)
    input_ids = []
    attention_masks = []
    token_type_ids = []
    max_len = max_len
    for sent in sentences:
        sent = sent.strip()
        sent = " ".join(sent.split())
        sent = ' ' + sent
        # print(sent)
        encoded_dict = tokenizer.encode_plus(sent,
                                                add_special_tokens=True,
                                                max_length=max_len, 
                                                pad_to_max_length=padding, 
                                                return_attention_mask = True,
                                                return_tensors = 'pt', 
                                                return_token_type_ids = use_type_tokens,
                                                truncation = True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        if use_type_tokens :
            token_type_ids.append(encoded_dict['token_type_ids'])
            # print("HELLO")
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if use_type_tokens :
        token_type_ids = torch.cat(token_type_ids, dim=0)

    
    #TODO: Pass dictionary instead of tuple
    if use_type_tokens :
        # print("input ids: {} attention_masks: {} token_type_ids: {}".format(input_ids.shape, attention_masks.shape, token_type_ids.shape))
        return {'input_ids':input_ids, 'attention_mask':attention_masks, 'token_type_ids':token_type_ids}

def bert_tokenize(sentences, max_seq_len=25, base='bert-base-uncased'):
	"""
	sentences : python list of sentences

	Returns :
		- bert corresponding tokens
	"""
	tokenizer = BertTokenizer.from_pretrained(base)

	# tokenize and encode the sentences
	tokens = tokenizer.batch_encode_plus(
	    sentences.tolist(),
	    max_length = max_seq_len,
	    pad_to_max_length=True,
	    truncation=True
	)

	return tokens

def roberta_tokenize(sentences, max_seq_len=25, base='roberta-base'):
    """
    sentences : python list of sentences

    Returns :
        - bert corresponding tokens
    """
    tokenizer = RobertaTokenizer.from_pretrained(base)

    # tokenize and encode the sentences
    tokens = tokenizer.batch_encode_plus(
        sentences.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True
    )

    return tokens

def plot_sentence_lengths(sentences):
	# get length of all the messages in the sentences
	seq_len = [len(i.split()) for i in sentences]
	pd.Series(seq_len).hist(bins = 30)
	print(max(seq_len))
