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
# For preprocessing ASCII...
from unidecode import unidecode

# specify GPU
# device = torch.device("cuda")

def summarise_data(data_path='data/english/v1/v1/covid19_disinfo_binary_english_train.tsv'):
    data = pd.read_csv(data_path, sep='\t')
    data=data.dropna(subset=['q7_label', 'q6_label'])
    tem = data.iloc[:, 2:].fillna('nan')
    print('Training Label Statistics')
    print('--------')
    print(tem.describe())
    print('--------')
    for i in range(7):
        print(i); print(tem['q'+str(i+1)+'_label'].value_counts()); print('-----');

def generate_class_weights(data_path='data/english/v1/v1/covid19_disinfo_binary_english_train.tsv'):
    print("Generating Class Weights")
    if not os.path.exists('data/class_weights'):
        os.mkdir('data/class_weights')

    data = pd.read_csv(data_path, sep='\t')
    data=data.dropna(subset=['q7_label', 'q6_label'])
    tem = data.iloc[:, 2:].fillna('nan')

    for i in range(7):
        x = np.array(tem['q' + str(i+1) + '_label'])
        wts = []
        class_count = len(np.where(x == 'no')[0])
        wts.append(class_count / (1.0 * len(x)))
        class_count = len(np.where(x == 'yes')[0])
        wts.append(class_count / (1.0 * len(x)))

        if i in [1, 2, 3, 4]:
            class_count = len(np.where(x == 'nan')[0])
            wts.append(class_count / (1.0 * len(x)))

        wts = np.array(wts)
        np.save(os.path.join('data/class_weights', 'q' + str(i+1) + '.npy'), wts)



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

def tokenize(sentences, use_type_tokens = True, padding = True, max_len = 25):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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

def bert_tokenize(sentences, max_seq_len=25, bert_base='bert-base-uncased'):
	"""
	sentences : python list of sentences

	Returns :
		- bert corresponding tokens
	"""
	tokenizer = BertTokenizerFast.from_pretrained(bert_base)

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
