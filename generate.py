import argparse
import math

import progressbar
import torch

from register import register
from train import shift_target_inputs_to_labels

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate(args):

    model_class, tokenizer_class, dataset_class = register(args.model_class)

    if args.score_reference:
        args.batch_size = 1
        test_dataset = dataset_class(
            tokenizer_class=tokenizer_class,
            tokenizer_path=args.save_dir,
            source_data_path=args.test_source_data_path,
            target_data_path=args.test_target_data_path
        )
    else:
        test_dataset = dataset_class(
            tokenizer_class=tokenizer_class,
            tokenizer_path=args.save_dir,
            source_data_path=args.test_source_data_path
        )
    test_dataloader = test_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    if args.src_lang is not None and args.tgt_lang is not None:
        src_lang_id = test_dataset.tokenizer.lang_code_to_id[args.src_lang]
        tgt_lang_id = test_dataset.tokenizer.lang_code_to_id[args.tgt_lang]

    model = model_class.from_pretrained(args.save_dir)
    model.to(DEVICE)
    model.eval()

    if not args.debug:
        num_batches = math.ceil(len(test_dataset) / args.batch_size)
        widgets = [
            progressbar.Percentage(), ' | ',
            progressbar.SimpleProgress(), ' ',
            progressbar.Bar('▇'), ' ',
            progressbar.Timer(), ' | ',
            progressbar.ETA()
        ]

        progress = progressbar.ProgressBar(
            max_value=num_batches,
            widgets=widgets,
            redirect_stdout=True
        ).start()

    output_file = open(args.output_path, 'w')

    for itr, data in enumerate(test_dataloader):

        if args.score_reference:
            src_input_ids, src_attn_mask, tgt_input_ids, tgt_attn_mask = (x.to(DEVICE) for x in data)
        else:
            src_input_ids, src_attn_mask = (x.to(DEVICE) for x in data)

        if args.score_reference:
            labels = shift_target_inputs_to_labels(tgt_input_ids, test_dataset.tokenizer.pad_token_id)
            with torch.no_grad():
                output = model(
                    src_input_ids,
                    attention_mask=src_attn_mask,
                    decoder_input_ids=tgt_input_ids,
                    decoder_attention_mask=tgt_attn_mask,
                    labels=labels
                )
            score = output[0].item()
            output_file.write(str(score) + '\n')
        else:
            with torch.no_grad():
                tgt_output_ids = model.generate(
                    src_input_ids,
                    attention_mask=src_attn_mask,
                    decoder_start_token_id = tgt_lang_id,
                    num_beams=args.beam_size,
                    num_return_sequences=args.num_return_sequences,
                    max_length=args.max_length
                )
            for seq_ids in tgt_output_ids.to('cpu').numpy().tolist():
                seq_toks = test_dataset.tokenizer.decode(
                    seq_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                output_file.write(seq_toks + '\n')

        if not args.debug:
            progress.update(itr+1)

    if not args.debug:
        progress.finish()

    output_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-class', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--test-source-data-path', type=str)
    parser.add_argument('--test-target-data-path', type=str, default=None)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=200)
    parser.add_argument('--score-reference', action='store_true')

    parser.add_argument('--src-lang', type=str, default=None)
    parser.add_argument('--tgt-lang', type=str, default=None)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    generate(parse_args())
