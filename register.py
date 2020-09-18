from transformers import (BartForConditionalGeneration, BartTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer,
                          MBartForConditionalGeneration, MBartTokenizer)
from dataset import Seq2SeqDataset, MBartSeq2SeqDataset

def register(model_name=None):

    if model_name in ['bart', 'facebook/bart-base', 'facebook/bart-large']:
        model_class = BartForConditionalGeneration
        tokenizer_class = BartTokenizer
        dataset_class = Seq2SeqDataset
    elif model_name in ['t5', 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
        model_class = T5ForConditionalGeneration
        tokenizer_class = T5Tokenizer
        dataset_class = Seq2SeqDataset
    elif model_name in ['mbart', 'facebook/mbart-large-cc25']:
        model_class = MBartForConditionalGeneration
        tokenizer_class = MBartTokenizer
        dataset_class = MBartSeq2SeqDataset
    else:
        raise Exception('Do not support this model')

    return model_class, tokenizer_class, dataset_class
