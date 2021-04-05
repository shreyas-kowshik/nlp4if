import argparse
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import copy
import time

parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="../data/english/v1/v1/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="../data/english/v2/v2/",
                    help="Expects a path to dev folder")
args = parser.parse_args()

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    # encoded = tokenizer.prepare_seq2seq_batch(src_texts)
    encoded = tokenizer.prepare_seq2seq_batch(src_texts,return_tensors="pt")
    
    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate(texts, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return back_translated_texts

def aug_sentence(text):
    augs = {}
    st=time.time()
    print(' Back Translating es')
    augs['es']=back_translate(text, source_lang="en", target_lang="es")
    print(time.time()-st)
    st=time.time()
    print(' Back Translating fr')
    augs['fr']=back_translate(text, source_lang="en", target_lang="fr")
    print(time.time()-st)
    st=time.time()
    print(' Back Translating de')
    augs['de']=back_translate(text, source_lang="en", target_lang="de")
    print(time.time()-st)
    return augs

TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

df_train = pd.read_csv(TRAIN_FILE, sep='\t')
df_train=df_train[:10]

target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)

augmented_dict = aug_sentence(df_train['tweet_text'].to_list())

df_train_es = copy.deepcopy(df_train)
df_train_fr = copy.deepcopy(df_train)
df_train_de = copy.deepcopy(df_train)

df_train_es.drop('tweet_text', inplace=True, axis=1)
df_train_fr.drop('tweet_text', inplace=True, axis=1)
df_train_de.drop('tweet_text', inplace=True, axis=1)

df_train_es['tweet_text']=augmented_dict['es']
df_train_fr['tweet_text']=augmented_dict['fr']
df_train_de['tweet_text']=augmented_dict['de']

df_train_es.to_csv('augmented_datasets/df_train_es_test.tsv', sep='\t', index=False)
df_train_fr.to_csv('augmented_datasets/df_train_fr_test.tsv', sep='\t', index=False)
df_train_de.to_csv('augmented_datasets/df_train_de_test.tsv', sep='\t', index=False)