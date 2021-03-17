import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

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
