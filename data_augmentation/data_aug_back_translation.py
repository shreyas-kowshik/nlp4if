import argparse
from transformers import MarianMTModel, MarianTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="data/english/v1/v1/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v2/v2/",
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

TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

df_train = pd.read_csv(TRAIN_FILE, sep='\t')

target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)

index = randint(0, len(df_train)-1)
en_texts = [df_train['tweet_text'][index]]

print("Original:")
print(en_texts)
print("en->es->en")
aug_texts = back_translate(en_texts, source_lang="en", target_lang="es")
print(aug_texts)
print("en->fr->en")
aug_texts = back_translate(en_texts, source_lang="en", target_lang="fr")
print(aug_texts)    


