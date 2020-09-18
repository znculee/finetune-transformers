import logging

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class Seq2SeqDataset(Dataset):

    def __init__(
        self,
        tokenizer_class,
        tokenizer_path,
        source_data_path,
        target_data_path=None,
        indivisible_tokens_path=None,
        cache_dir=None,
        save_tokenizer=None
    ):
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, cache_dir=cache_dir)

        with open(source_data_path) as f:
            self.source = f.readlines()
        if target_data_path:
            with open(target_data_path) as f:
                self.target = f.readlines()
        else:
            self.target = None

        self._add_indivisible_tokens(indivisible_tokens_path)
        self.vocab_size = self._get_vocab_size()

        if save_tokenizer:
            self.tokenizer.save_pretrained(save_tokenizer)

    def _add_indivisible_tokens(self, indivisible_tokens_path):
        if indivisible_tokens_path is not None:
            logger.info('adding indivisible tokens to the vocabulary')
            with open(indivisible_tokens_path, 'r') as f:
                indivisible_tokens = [l.strip() for l in f.readlines()]
            self.tokenizer.add_tokens(indivisible_tokens)

    def _get_vocab_size(self):
        return len(self.tokenizer)

    def __len__(self):
        if self.target:
            assert len(self.source) == len(self.target)
        return len(self.source)

    def __getitem__(self, index):
        if self.target:
            item = (self.source[index], self.target[index])
        else:
            item = self.source[index]
        return item

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        """
        Overload the method of adding special tokens
        e.g.
        BART: <bos_id> token_ids_0 <eos_id>
        T5:   <pad_id> token_ids_0 <eos_id>
        """
        if self.tokenizer.bos_token_id is None:
            prefix_tokens = [self.tokenizer.pad_token_id]
        else:
            prefix_tokens = [self.tokenizer.bos_token_id]
        suffix_tokens = [self.tokenizer.eos_token_id]
        if token_ids_1 is None:
            return prefix_tokens + token_ids_0 + suffix_tokens
        else:
            raise Exception('Don\'t expect to tokenize pairs')

    def _collate_fn(self, data):
        source_list = []
        if self.target:
            target_list = []
        for item in data:
            if self.target:
                source, target = item
            else:
                source = item
            source_list.append(source.strip())
            if self.target:
                target_list.append(target.strip())

        self.tokenizer.build_inputs_with_special_tokens = self._build_inputs_with_special_tokens
        source_batch = self.tokenizer(
            source_list,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )
        if self.target:
            target_batch = self.tokenizer(
                target_list,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt'
            )

        if self.target:
            collated_data = (
                source_batch['input_ids'],
                source_batch['attention_mask'],
                target_batch['input_ids'],
                target_batch['attention_mask']
            )
        else:
            collated_data = (
                source_batch['input_ids'],
                source_batch['attention_mask']
            )

        return collated_data

    def get_dataloader(self, batch_size, shuffle):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )


class MBartSeq2SeqDataset(Seq2SeqDataset):

    def __init__(self, src_lang='en_XX', tgt_lang='en_XX', **kwargs):
        super().__init__(**kwargs)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def _add_indivisible_tokens(self, indivisible_tokens_path):
        if indivisible_tokens_path is not None:
            logger.info('adding indivisible tokens to the vocabulary')
            with open(indivisible_tokens_path, 'r') as f:
                indivisible_tokens = [l.strip() for l in f.readlines()]
            vocab_size = self._get_vocab_size()
            indivisible_tok_to_id = {tok: vocab_size + i for i, tok in enumerate(indivisible_tokens)}
            self.tokenizer.fairseq_tokens_to_ids.update(indivisible_tok_to_id)
            self.tokenizer.fairseq_ids_to_tokens = {v: k for k, v in self.tokenizer.fairseq_tokens_to_ids.items()}
            self.tokenizer._additional_special_tokens.extend(indivisible_tokens)
            self.tokenizer.unique_no_split_tokens.extend(indivisible_tokens)

    def _get_vocab_size(self):
        return max(self.tokenizer.fairseq_ids_to_tokens.keys()) + 1

    def _collate_fn(self, data):
        source_list = []
        if self.target:
            target_list = []
        for item in data:
            if self.target:
                source, target = item
            else:
                source = item
            source_list.append(source.strip())
            if self.target:
                target_list.append(target.strip())

        self.tokenizer.cur_lang_code = self.tokenizer.lang_code_to_id[self.src_lang]
        self.tokenizer.prefix_tokens = []
        self.tokenizer.suffix_tokens = [self.tokenizer.eos_token_id, self.tokenizer.cur_lang_code]
        source_batch = self.tokenizer(
            source_list,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )
        if self.target:
            self.tokenizer.cur_lang_code = self.tokenizer.lang_code_to_id[self.tgt_lang]
            self.tokenizer.prefix_tokens = [self.tokenizer.cur_lang_code]
            self.tokenizer.suffix_tokens = [self.tokenizer.eos_token_id]
            target_batch = self.tokenizer(
                target_list,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt'
            )

        if self.target:
            collated_data = (
                source_batch['input_ids'],
                source_batch['attention_mask'],
                target_batch['input_ids'],
                target_batch['attention_mask']
            )
        else:
            collated_data = (
                source_batch['input_ids'],
                source_batch['attention_mask']
            )

        return collated_data
