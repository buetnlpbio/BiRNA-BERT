Authors' implementation of **BiRNA-BERT Allows Efficient RNA Language Modeling with Adaptive Tokenization**

🤗[BiRNA-BERT Model Zoo](https://huggingface.co/collections/buetnlpbio/birna-bert-66840c2645d8ceb446b6c919)
- [Paper Link](https://www.biorxiv.org/content/10.1101/2024.07.02.601703v1)
## BiRNA-BERT

BiRNA-BERT is a BERT-style transformer encoder model that generates embeddings for RNA sequences. BiRNA-BERT has been trained on BPE tokens and individual nucleotides. As a result, it can generate both granular nucleotide-level embeddings and efficient sequence-level embeddings (using BPE).

BiRNA-BERT was trained using the MosaicBERT framework - https://huggingface.co/mosaicml/mosaic-bert-base


## Extracting RNA embeddings

```python
import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("buetnlpbio/birna-tokenizer")

config = transformers.BertConfig.from_pretrained("buetnlpbio/birna-bert")
mysterybert = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert",config=config,trust_remote_code=True)
mysterybert.cls = torch.nn.Identity()

# To get sequence embeddings
seq_embed = mysterybert(**tokenizer("AGCTACGTACGT", return_tensors="pt"))
print(seq_embed.logits.shape) # CLS + 4 BPE token embeddings + SEP

# To get nucleotide embeddings
char_embed = mysterybert(**tokenizer("A G C T A C G T A C G T", return_tensors="pt")) 
print(char_embed.logits.shape) # CLS + 12 nucleotide token embeddings + SEP
```

## Explicitly increasing max sequence length

```python
config = transformers.BertConfig.from_pretrained("buetnlpbio/birna-bert")
config.alibi_starting_size = 2048 # maximum sequence length updated to 2048 from config default of 1024

mysterybert = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert",config=config,trust_remote_code=True)
```

## Download Model and Tokenizer (External Link)

[Download model](https://file.io/EdXPvXfFBNU5) <br>
[Download tokenizer](https://file.io/dPlREMjAuDBs)

