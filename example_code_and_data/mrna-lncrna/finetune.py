import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_constant_schedule_with_warmup
from tqdm import tqdm
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)  # if using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


MODEL_NAME = "buetnlpbio/birna-bert"
TOKENIZER = "buetnlpbio/birna-tokenizer"
LR = 5e-4
BATCH_SIZE = 32
DEVICE = "cuda"
WARMUP_RATIO = 0.1
EPOCHS = 5
DATASET = "gma-ath"


import pandas as pd
df_ath = pd.read_csv('data/fasta_ath/PmliPred_label_ath.csv')
df_gma = pd.read_csv('data/fasta_gma/PmliPred_label_gma.csv')
df_mtr = pd.read_csv('data/fasta_mtr/PmliPred_label_mtr.csv')


class interaction_pairs:
    def __init__(self, mirna, lncrna):
        self.mirna = mirna
        self.lncrna = lncrna

    def __eq__(self, other):
        if isinstance(other, interaction_pairs):
            return self.mirna == other.mirna and self.lncrna == other.lncrna
        return False

    def __hash__(self):
        return hash((self.mirna, self.lncrna))


# create a dictionary with keys as the interaction pairs and values as the interaction scores
interaction_dict = {}
for i in range(0, len(df_ath)):
    mirna = df_ath.iloc[i]['A']
    lncrna = df_ath.iloc[i]['B']
    interaction_dict[interaction_pairs(mirna, lncrna)] = df_ath.iloc[i]['Label']

for i in range(0, len(df_gma)):
    mirna = df_gma.iloc[i]['A']
    lncrna = df_gma.iloc[i]['B']
    interaction_dict[interaction_pairs(mirna, lncrna)] = df_gma.iloc[i]['Label']

for i in range(0, len(df_mtr)):
    mirna = df_mtr.iloc[i]['A']
    lncrna = df_mtr.iloc[i]['B']
    interaction_dict[interaction_pairs(mirna, lncrna)] = df_mtr.iloc[i]['Label']

sequences_ath_lncrna=[]
sequences_ath_mirna=[]
sequences_gma_lncrna=[]
sequences_gma_mirna=[]
sequences_mtr_lncrna=[]
sequences_mtr_mirna=[]

ids_ath_lncrna=[]
ids_ath_mirna=[]
ids_gma_lncrna=[]
ids_gma_mirna=[]
ids_mtr_lncrna=[]
ids_mtr_mirna=[]


with open('data/fasta_ath/lncrna.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids_ath_lncrna.append(line.strip())
        else:
            sequences_ath_lncrna.append(line.strip())

with open('data/fasta_ath/mirna.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids_ath_mirna.append(line.strip())
        else:
            sequences_ath_mirna.append(line.strip())

with open('data/fasta_gma/lncrna.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids_gma_lncrna.append(line.strip())
        else:
            sequences_gma_lncrna.append(line.strip())

with open('data/fasta_gma/mirna.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids_gma_mirna.append(line.strip())
        else:
            sequences_gma_mirna.append(line.strip())

with open('data/fasta_mtr/lncrna.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids_mtr_lncrna.append(line.strip())
        else:
            sequences_mtr_lncrna.append(line.strip())

with open('data/fasta_mtr/mirna.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids_mtr_mirna.append(line.strip())
        else:
            sequences_mtr_mirna.append(line.strip())


ath_dataset_pairs = []
ath_dataset_labels = []
gma_dataset_pairs = []
gma_dataset_labels = []
mtr_dataset_pairs = []
mtr_dataset_labels = []


for i in range(0, len(ids_ath_mirna)):
    for j in range(0, len(ids_ath_lncrna)):

        interaction_object = interaction_pairs(ids_ath_mirna[i], ids_ath_lncrna[j])
        if interaction_object in interaction_dict:
            ath_dataset_pairs.append([sequences_ath_mirna[i], sequences_ath_lncrna[j]])
            ath_dataset_labels.append(interaction_dict[interaction_object])

for i in range(0, len(ids_gma_mirna)):
    for j in range(0, len(ids_gma_lncrna)):

        interaction_object = interaction_pairs(ids_gma_mirna[i], ids_gma_lncrna[j])
        if interaction_object in interaction_dict:
            gma_dataset_pairs.append([sequences_gma_mirna[i], sequences_gma_lncrna[j]])
            gma_dataset_labels.append(interaction_dict[interaction_object])

for i in range(0, len(ids_mtr_mirna)):
    for j in range(0, len(ids_mtr_lncrna)):

        interaction_object = interaction_pairs(ids_mtr_mirna[i], ids_mtr_lncrna[j])
        if interaction_object in interaction_dict:
            mtr_dataset_pairs.append([sequences_mtr_mirna[i], sequences_mtr_lncrna[j]])
            mtr_dataset_labels.append(interaction_dict[interaction_object])


# Check if CUDA is available and set device to GPU if it is
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


config = transformers.BertConfig.from_pretrained(MODEL_NAME) # the config needs to be passed
birnabert_mirna = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,config=config, trust_remote_code=True)
birnabert_mirna.cls = torch.nn.Identity()
birnabert_mirna.to(device)
birnabert_lncrna = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,config=config, trust_remote_code=True)
birnabert_lncrna.cls = torch.nn.Identity()
birnabert_lncrna.to(device)

for param in birnabert_mirna.parameters():
    param.requires_grad = False

for param in birnabert_lncrna.parameters():
    param.requires_grad = False

class RNA_RNA(Dataset):
    def __init__(self, list, labels):
        self.list_mirna, self.list_lncrna = zip(*list)

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq_mirna= self.list_mirna[idx].replace('U', 'T')

        #nucleotide tokenization for short sequences
        seq_mirna = " ".join(seq_mirna)
        #bpe tokenization for long sequences
        seq_lncrna = self.list_lncrna[idx].replace('U', 'T')

        # seq_lncrna = seq_lncrna[:1022]
        # seq_lncrna = " ".join(seq_lncrna)


        tokenized_output_mirna = tokenizer(seq_mirna, return_tensors="pt")
        tokenized_output_mirna = tokenized_output_mirna.to(device)

        tokenized_output_lncrna = tokenizer(seq_lncrna, return_tensors="pt")
        tokenized_output_lncrna = tokenized_output_lncrna.to(device)

        input_ids_mirna = tokenized_output_mirna['input_ids'].squeeze(0)
        input_ids_lncrna = tokenized_output_lncrna['input_ids'].squeeze(0)


        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.int64).to(device)

        return  (input_ids_mirna,input_ids_lncrna), label_tensor


def get_list_seperated(list):
    list1 = []
    list2 = []
    for i in range(len(list)):
        list1.append(list[i][0])
        list2.append(list[i][1])
    return list1, list2


def custom_collate_fn(batch):
    # Unzip the batch
    inputs, labels = zip(*batch)
    input_ids_mirna, input_ids_lncrna = zip(*inputs)

    input_ids_mirna_padded = pad_sequence(input_ids_mirna, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids_lncrna_padded = pad_sequence(input_ids_lncrna, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_mask_mirna = (input_ids_mirna_padded != tokenizer.pad_token_id).float()
    attention_mask_lncrna = (input_ids_lncrna_padded != tokenizer.pad_token_id).float()

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.float16)

    return (input_ids_mirna_padded, attention_mask_mirna, input_ids_lncrna_padded, attention_mask_lncrna), labels

# When creating DataLoaders, add worker_init_fn and generator
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

list_mirna, list_lncrna = get_list_seperated(ath_dataset_pairs)
ath_dataset = RNA_RNA(list(zip(list_mirna, list_lncrna)), ath_dataset_labels)
ath_dataloader = DataLoader(ath_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, generator=g, worker_init_fn=seed_worker)

list_mirna, list_lncrna = get_list_seperated(gma_dataset_pairs)
gma_dataset = RNA_RNA(list(zip(list_mirna, list_lncrna)), gma_dataset_labels)
gma_dataloader = DataLoader(gma_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, generator=g, worker_init_fn=seed_worker)


list_mirna, list_lncrna = get_list_seperated(mtr_dataset_pairs)
mtr_dataset = RNA_RNA(list(zip(list_mirna, list_lncrna)), mtr_dataset_labels)
mtr_dataloader = DataLoader(mtr_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, generator=g,worker_init_fn=seed_worker)



class DualBertClassifier(nn.Module):
    def __init__(self):
        super(DualBertClassifier, self).__init__()

        # Initialize two separate BERT models

        self.bert_mirna = birnabert_mirna
        self.bert_lncrna = birnabert_lncrna

        # Define convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)

        # Define fully connected layers
        self.fc1 = nn.Linear(195840, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, input_ids_mirna, attention_mask_mirna, input_ids_lncrna, attention_mask_lncrna):
        # Process sequences through respective BERT models
        with torch.no_grad():
            outputs_mirna = self.bert_mirna(input_ids_mirna, attention_mask=attention_mask_mirna).logits
            outputs_lncrna = self.bert_lncrna(input_ids_lncrna, attention_mask=attention_mask_lncrna).logits

        pooled_mirna = torch.mean(outputs_mirna, dim=1)
        pooled_lncrna = torch.mean(outputs_lncrna, dim=1)
        # Pass through convolutional layers
        x = torch.cat((pooled_mirna, pooled_lncrna), dim=1)
        x = x.unsqueeze(1)
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.conv3(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x).squeeze(1)
        return x

model = DualBertClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCELoss()

if DATASET == "ath-gma":
    train_loader = ath_dataloader
    test_loader = gma_dataloader
elif DATASET == "ath-mtr":
    train_loader = ath_dataloader
    test_loader = mtr_dataloader
elif DATASET == "gma-mtr":
    train_loader = gma_dataloader
    test_loader = mtr_dataloader
elif DATASET == "gma-ath":
    train_loader = gma_dataloader
    test_loader = ath_dataloader
elif DATASET == "mtr-ath":
    train_loader = mtr_dataloader
    test_loader = ath_dataloader
elif DATASET == "mtr-gma":
    train_loader = mtr_dataloader
    test_loader = gma_dataloader



num_epochs = EPOCHS
num_steps = len(train_loader) * num_epochs
num_warmup_steps = int(WARMUP_RATIO * num_steps)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)


def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Initialize bins for length-based accuracy
    length_bins = {
        '1-500': {'correct': 0, 'total': 0},
        '501-1000': {'correct': 0, 'total': 0},
        '1001-1500': {'correct': 0, 'total': 0},
        '1501-2000': {'correct': 0, 'total': 0},
        '2001-2500': {'correct': 0, 'total': 0},
        '2501-3000': {'correct': 0, 'total': 0},
        '3001-3500': {'correct': 0, 'total': 0},
        '3501-4000': {'correct': 0, 'total': 0},
        '4000+': {'correct': 0, 'total': 0}
        
    }

    with torch.no_grad():
        for (input_ids_mirna, attention_mask_mirna, input_ids_lncrna, attention_mask_lncrna), labels in tqdm(dataloader):
            input_ids_mirna = input_ids_mirna.to(device).long()
            input_ids_lncrna = input_ids_lncrna.to(device).long()
            attention_mask_mirna = attention_mask_mirna.to(device).float()
            attention_mask_lncrna = attention_mask_lncrna.to(device).float()

            labels = labels.to(device).float()

            logits = model(input_ids_mirna, attention_mask_mirna, input_ids_lncrna, attention_mask_lncrna)

            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = (logits > 0.5).float()  # Convert probabilities to binary output
            batch_correct = (predicted_labels == labels).float()
            total_correct += batch_correct.sum().item()
            total_samples += labels.size(0)            
            # Decode input IDs to get actual sequence lengths
            decoded_sequences = tokenizer.batch_decode(input_ids_lncrna, skip_special_tokens=True)
            actual_lengths = [len(seq.replace(" ", "")) for seq in decoded_sequences]
            
            # Bin the accuracies by sequence length
            for i, length in enumerate(actual_lengths):
                if length <= 500:
                    bin_key = '1-500'
                elif length <= 1000:
                    bin_key = '501-1000'
                elif length <= 1500:
                    bin_key = '1001-1500'
                elif length <= 2000:
                    bin_key = '1501-2000'
                elif length <= 2500:
                    bin_key = '2001-2500'
                elif length <= 3000:
                    bin_key = '2501-3000'
                elif length <= 3500:
                    bin_key = '3001-3500'
                elif length <= 4000:
                    bin_key = '3501-4000'
                else:
                    bin_key = '4000+'
                    
                length_bins[bin_key]['correct'] += batch_correct[i].item()
                length_bins[bin_key]['total'] += 1

    average_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    return average_loss, accuracy, length_bins



for epoch_count in range(1,num_epochs+1):

    # Training loop
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, ((input_ids_mirna, attention_mask_mirna, input_ids_lncrna, attention_mask_lncrna), labels) in enumerate(tqdm(train_loader)):
        input_ids_mirna = input_ids_mirna.to(device).long()
        input_ids_lncrna = input_ids_lncrna.to(device).long()
        attention_mask_mirna = attention_mask_mirna.to(device).float()
        attention_mask_lncrna = attention_mask_lncrna.to(device).float()

        labels = labels.to(device).float()

        logits = model(input_ids_mirna, attention_mask_mirna, input_ids_lncrna, attention_mask_lncrna)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        predicted_labels = (logits > 0.5).float()
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    print(f'Epoch {epoch_count}, Training Loss: {average_loss}, Training Accuracy: {accuracy * 100:.2f}%')


    test_loss, test_accuracy, length_bins = validate(model, test_loader, criterion, device)
    print(f'Epoch {epoch_count}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%')

    from prettytable import PrettyTable
    
    length_accuracies = {}
    table = PrettyTable()
    table.field_names = ["Bin", "Total", "Correct", "Accuracy"]
    
    for bin_key, counts in length_bins.items():
        if counts['total'] > 0:
            accuracy = counts['correct'] / counts['total']
            table.add_row([
                bin_key,
                counts['total'],
                counts['correct'],
                f"{accuracy * 100:.2f}%"
            ])
            length_accuracies[bin_key] = counts['correct'] / counts['total']
        else:
            table.add_row([bin_key, 0, 0, "0.00%"])
            length_accuracies[bin_key] = 0.0
            
    print(table)
            
