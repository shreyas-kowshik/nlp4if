import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, BertTokenizer

# specify GPU
# device = torch.device("cuda")

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
    sentences = data["tweet_text"]
    labels = np.array(data.iloc[:, 2:].fillna('nan'))

    from sklearn import preprocessing
    les = []
    for i in range(labels.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(labels[:, i])
        labels[:, i] = le.transform(labels[:, i])
        les.append(le)
        
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

def bert_tokenize(sentences, max_seq_len=25):
	"""
	sentences : python list of sentences

	Returns :
		- bert corresponding tokens
	"""
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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
