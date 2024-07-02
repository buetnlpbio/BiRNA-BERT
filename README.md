## Download Model and Tokenizer

[Download model](https://file.io/EdXPvXfFBNU5) <br>
[Download tokenizer](https://file.io/dPlREMjAuDBs)


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
