# Code for fine-tuning [huggingface/transformers](https://github.com/huggingface/transformers)

## Setup

[huggingface/transformers](https://github.com/huggingface/transformers) should be [installed from the source](https://huggingface.co/transformers/installation.html#installing-from-source).
The code has been tested on commit `3babef81` of [huggingface/transformers](https://github.com/huggingface/transformers).

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout -b webnlg e9014fb
pip install -e .
cd ..
git clone https://github.com/znculee/finetune-transformers.git
```
