Authors' implementation of **BiRNA-BERT Allows Efficient RNA Language Modeling with Adaptive Tokenization**

🤗[BiRNA-BERT Model Zoo](https://huggingface.co/collections/buetnlpbio/birna-bert-66840c2645d8ceb446b6c919)
- [Paper Link](https://www.biorxiv.org/content/10.1101/2024.07.02.601703v1)
## BiRNA-BERT

BiRNA-BERT is a BERT-style transformer encoder model that generates embeddings for RNA sequences. BiRNA-BERT has been trained on BPE tokens and individual nucleotides. As a result, it can generate both granular nucleotide-level embeddings and efficient sequence-level embeddings (using BPE).

BiRNA-BERT was trained using the MosaicBERT framework - https://huggingface.co/mosaicml/mosaic-bert-base


## Extracting RNA embeddings

```python
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 50
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 1
MODEL_NAME = "Lancelot53/birnabert-2ep"
TOKENIZER = "buetnlpbio/birna-tokenizer"

# https://huggingface.co/Lancelot53/birnabert-2ep
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import transformers
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup, AutoModelForMaskedLM, AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
config = transformers.BertConfig.from_pretrained(MODEL_NAME)
birnabert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,config=config,trust_remote_code=True)
birnabert.cls = torch.nn.Identity()

birnabert.to(device)

# To get sequence embeddings
seq_embed = birnabert(**tokenizer("AGCTACGTACGT", return_tensors="pt"))
print(seq_embed.logits.shape) # CLS + 4 BPE token embeddings + SEP

# To get nucleotide embeddings
char_embed = birnabert(**tokenizer("A G C T A C G T A C G T", return_tensors="pt")) 
print(char_embed.logits.shape) # CLS + 12 nucleotide token embeddings + SEP
```

## Explicitly increasing max sequence length

```python
config = transformers.BertConfig.from_pretrained(MODEL_NAME)
config.alibi_starting_size = 2048 # maximum sequence length updated to 2048 from config default of 1024

mysterybert = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert",config=config,trust_remote_code=True)
```

## Download Model and Tokenizer (External Link)

[Download model](https://file.io/EdXPvXfFBNU5) <br>
[Download tokenizer](https://file.io/dPlREMjAuDBs)

## Citation
```
@article {Tahmid2024.07.02.601703,
	author = {Tahmid, Md Toki and Shahgir, Haz Sameen and Mahbub, Sazan and Dong, Yue and Bayzid, Md. Shamsuzzoha},
	title = {BiRNA-BERT Allows Efficient RNA Language Modeling with Adaptive Tokenization},
	elocation-id = {2024.07.02.601703},
	year = {2024},
	doi = {10.1101/2024.07.02.601703},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/07/04/2024.07.02.601703},
	eprint = {https://www.biorxiv.org/content/early/2024/07/04/2024.07.02.601703.full.pdf},
	journal = {bioRxiv}
}
```
