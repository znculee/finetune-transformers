import argparse
import logging
import math

import progressbar
import sacrebleu
import torch
import torch.optim as optim

from register import register

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

def shift_target_inputs_to_labels(tgt_input_ids, pad_token_id):
    """
    <bos> word1 word2 word3 <eos> (target input)
    word1 word2 word3 <eos> <pad> (target label)
    """
    batch_pads = torch.empty(
        tgt_input_ids.shape[0], 1,
        dtype=tgt_input_ids.dtype,
        device=DEVICE
    ).fill_(pad_token_id)
    labels = torch.cat((tgt_input_ids[:, 1:], batch_pads), dim=1)
    return labels

def train(args):

    logfile = logging.FileHandler(args.save_dir + '/log.txt', mode='w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    model_class, tokenizer_class, dataset_class = register(args.pretrained_model_path)

    train_dataset = dataset_class(
        tokenizer_class=tokenizer_class,
        tokenizer_path=args.pretrained_model_path,
        source_data_path=args.train_source_data_path,
        target_data_path=args.train_target_data_path,
        indivisible_tokens_path=args.indivisible_tokens_path,
        cache_dir=args.cache_dir,
        save_tokenizer=args.save_dir
    )
    train_dataloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    valid_dataset = dataset_class(
        tokenizer_class=tokenizer_class,
        tokenizer_path=args.save_dir,
        source_data_path=args.valid_source_data_path,
        target_data_path=args.valid_target_data_path
    )
    valid_dataloader = valid_dataset.get_dataloader(batch_size=args.valid_batch_size, shuffle=False)

    if args.src_lang is not None and args.tgt_lang is not None:
        src_lang_id = train_dataset.tokenizer.lang_code_to_id[args.src_lang]
        tgt_lang_id = train_dataset.tokenizer.lang_code_to_id[args.tgt_lang]

    model = model_class.from_pretrained(args.pretrained_model_path, cache_dir=args.cache_dir)
    if args.indivisible_tokens_path is not None:
        model.resize_token_embeddings(train_dataset.vocab_size)
    model.to(DEVICE)
    model.train()
    logger.info(f'model\n{model}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'total parameters: {num_total_params}')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    logger.info(f'optimizer\n{optimizer}')

    if not args.debug:
        train_num_batchs_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
        train_progress_widgets = [
            progressbar.Percentage(), ' | ',
            progressbar.SimpleProgress(), ' | ',
            progressbar.Variable('step', width=0), ' | ',
            progressbar.Variable('loss', width=0, precision=6), ' ',
            progressbar.Bar('▇'), ' ',
            progressbar.Timer(), ' | ',
            progressbar.ETA()
        ]
        valid_num_batchs_per_epoch = math.ceil(len(valid_dataset) / args.valid_batch_size)
        valid_progress_widgets = [
            progressbar.Percentage(), ' | ',
            progressbar.SimpleProgress(), ' ',
            progressbar.Bar('▇'), ' ',
            progressbar.Timer(), ' | ',
            progressbar.ETA()
        ]

    global_step = 1
    best_valid_measure = math.inf
    best_epoch_itr = 0

    for epoch_itr in range(args.max_epoch):

        train_epoch_sum_loss = 0
        train_epoch_average_loss = 0

        logger.info(f'begin training epoch {epoch_itr+1}')
        if not args.debug:
            train_progress = progressbar.ProgressBar(
                max_value=train_num_batchs_per_epoch,
                widgets=train_progress_widgets,
                redirect_stdout=True
            ).start()

        for itr, data in enumerate(train_dataloader):

            src_input_ids, src_attn_mask, tgt_input_ids, tgt_attn_mask = (x.to(DEVICE) for x in data)

            labels = shift_target_inputs_to_labels(tgt_input_ids, train_dataset.tokenizer.pad_token_id)

            output = model(
                input_ids=src_input_ids,
                attention_mask=src_attn_mask,
                decoder_input_ids=tgt_input_ids,
                decoder_attention_mask=tgt_attn_mask,
                labels=labels
            )

            loss = output[0]
            train_epoch_sum_loss += loss * src_input_ids.shape[0]

            normalized_loss = loss / args.update_frequency
            normalized_loss.backward()

            global_step += 1
            if not args.debug:
                train_progress.update(itr+1, step=global_step, loss=loss)

            if (itr + 1) % args.update_frequency == 0:
                optimizer.step()
                optimizer.zero_grad()

        if not args.debug:
            train_progress.finish()

        train_epoch_average_loss = train_epoch_sum_loss.item() / len(train_dataset)
        logger.info(f'average training loss: {train_epoch_average_loss}')

        logger.info(f'begin validation for epoch {epoch_itr+1}')
        model.eval()

        if not args.debug:
            valid_progress = progressbar.ProgressBar(
                max_value=valid_num_batchs_per_epoch,
                widgets=valid_progress_widgets,
                redirect_stdout=True
            ).start()

        valid_measure = 0
        if args.valid_bleu:
            hypotheses = []
            references = []
        else:
            valid_epoch_sum_loss = 0

        for itr, data in enumerate(valid_dataloader):

            src_input_ids, src_attn_mask, tgt_input_ids, tgt_attn_mask = (x.to(DEVICE) for x in data)

            if args.valid_bleu:
                with torch.no_grad():
                    tgt_output_ids = model.generate(
                        src_input_ids,
                        attention_mask=src_attn_mask,
                        decoder_start_token_id = tgt_lang_id,
                        num_beams=args.valid_beam_size,
                        max_length=args.valid_max_length
                    )
                for seq_ids in tgt_output_ids.to('cpu').numpy().tolist():
                    seq_toks = valid_dataset.tokenizer.decode(
                        seq_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    hypotheses.append(seq_toks)
                for seq_ids in tgt_input_ids.to('cpu').numpy().tolist():
                    seq_toks = valid_dataset.tokenizer.decode(
                        seq_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    references.append(seq_toks)
            else:
                labels = shift_target_inputs_to_labels(tgt_input_ids, valid_dataset.tokenizer.pad_token_id)
                with torch.no_grad():
                    output = model(
                        input_ids=src_input_ids,
                        attention_mask=src_attn_mask,
                        decoder_input_ids=tgt_input_ids,
                        decoder_attention_mask=tgt_attn_mask,
                        labels=labels
                    )
                valid_loss = output[0]
                valid_epoch_sum_loss += valid_loss * src_input_ids.shape[0]

            if not args.debug:
                valid_progress.update(itr+1)

        model.train()
        if not args.debug:
            valid_progress.finish()

        if args.valid_bleu:
            bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)
            valid_measure = -bleu.score
            logger.info(f'validation BLEU: {bleu.score}')
        else:
            valid_measure = valid_epoch_sum_loss.item() / len(valid_dataset)
            logger.info(f'validation loss: {valid_measure}')

        if valid_measure < best_valid_measure:
            logger.info('saving new best checkpoints')
            best_valid_measure = valid_measure
            best_epoch_itr = epoch_itr + 1
            model.save_pretrained(args.save_dir)

        if (epoch_itr + 1 - best_epoch_itr) > args.patience:
            logger.info(f'early stop since valid performance hasn\'t improved for last {args.patience} eopchs')
            break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model-path', type=str)
    parser.add_argument('--train-source-data-path', type=str)
    parser.add_argument('--train-target-data-path', type=str)
    parser.add_argument('--valid-source-data-path', type=str)
    parser.add_argument('--valid-target-data-path', type=str)
    parser.add_argument('--indivisible-tokens-path', type=str, default=None)

    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--cache-dir', type=str)

    parser.add_argument('--max-epoch', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--update-frequency', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--valid-batch-size', type=int, default=8)
    parser.add_argument('--valid-bleu', action='store_true')
    parser.add_argument('--valid-beam-size', type=int, default=5)
    parser.add_argument('--valid-max-length', type=int, default=200)

    parser.add_argument('--src-lang', type=str, default=None)
    parser.add_argument('--tgt-lang', type=str, default=None)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train(parse_args())
