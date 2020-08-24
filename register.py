from transformers import (BartForConditionalGeneration, BartTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

def register(model_name=None):

    if model_name in ['bart', 'facebook/bart-base', 'facebook/bart-large']:
        model_class = BartForConditionalGeneration
        tokenizer_class = BartTokenizer
    elif model_name in ['t5', 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
        model_class = T5ForConditionalGeneration
        tokenizer_class = T5Tokenizer
    else:
        raise Exception('Do not support this model')

    return model_class, tokenizer_class
