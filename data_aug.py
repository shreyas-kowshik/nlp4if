import nlpaug.augmenter.word as naw
import nlpaug.flow as nafc
from nlpaug.util import Action
from random import randint

aug_ins = naw.ContextualWordEmbsAug(
    model_path='roberta-large', action="insert")
aug_sub = naw.ContextualWordEmbsAug(
    model_path='roberta-large', action="substitute")

index = randint(0, len(df_train)-1)
text = df_train['tweet_text'][index]

print("Original:")
print(text)
print("Augmented (insert) Text:")
print(aug_ins.augment(text))
print("Augmented (subs) Text:")
print(aug_sub.augment(text))    

print('[WIP] back translation model to large')
do=False
if do:
	back_translation_aug = naw.BackTranslationAug(
	    from_model_name='transformer.wmt19.en-de', 
	    to_model_name='transformer.wmt19.de-en'
	)
	print("Augmented (back translation) Text:")
	print(back_translation_aug.augment(text))