import argparse
import math

import progressbar
import torch

from dataset import Seq2SeqDataset
from register import register

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate(args):

    model_class, tokenizer_class = register(args.model_class)

    test_dataset = Seq2SeqDataset(
        tokenizer_class=tokenizer_class,
        tokenizer_path=args.save_dir,
        source_data_path=args.test_source_data_path
    )
    test_dataloader = test_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    model = model_class.from_pretrained(args.save_dir)
    model.to(DEVICE)
    model.eval()

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

        src_input_ids, src_attn_mask = (x.to(DEVICE) for x in data)

        with torch.no_grad():
            tgt_output_ids = model.generate(
                src_input_ids,
                attention_mask=src_attn_mask,
                num_beams=args.beam_size,
                max_length=args.max_length
            )

        for seq_ids in tgt_output_ids.to('cpu').numpy().tolist():
            seq_toks = test_dataset.tokenizer.decode(
                seq_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            output_file.write(seq_toks + '\n')

        progress.update(itr+1)

    progress.finish()

    output_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-class', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--test-source-data-path', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=200)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    generate(parse_args())
