import numpy as np  
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_constant_schedule_with_warmup
torch.manual_seed(42)

LR = 1e-6
BATCH_SIZE = 64
EPOCHS = 10
WARMUP_RATIO = 0.1

MODEL_NAME = "../../MODEL"
TOKENIZER = "../../TOKENIZER"
dataset_name = "data/AKAP1/AKAP1"
MAX_TOKENS = 1022
k = 1 #tradeoff
ONLY_BPE = False
ONLY_NUC = False

def dynamic_tokenize_preprocessing(seq):
    if ONLY_NUC:
        return " ".join(seq)
    elif ONLY_BPE:
        return seq
    
    if len(seq) < k*MAX_TOKENS:
        seq = " ".join(seq[:MAX_TOKENS])
    return seq

test_file_x=torch.load(f"{dataset_name}_test_x_tensors")
valid_file_x=torch.load(f"{dataset_name}_valid_x_tensors")
train_file_x=torch.load(f"{dataset_name}_train_x_tensors")
label_test=np.loadtxt(f"{dataset_name}_test_y")
label_valid=np.loadtxt(f"{dataset_name}_valid_y")
label_train=np.loadtxt(f"{dataset_name}_train_y")

train_seqs=[]
valid_seqs=[]
test_seqs=[]

for i in range(len(train_file_x)):
    temp_seq= train_file_x[i].numpy()
    temp_seq = temp_seq[1:-1]
    # convert 5 to a, 6 to u, 7 to c, 8 to g

    temp_str = ""
    for j in range(len(temp_seq)):
        if temp_seq[j] == 5:
            temp_str += "A"
        elif temp_seq[j] == 6:
            temp_str += "T"
        elif temp_seq[j] == 7:
            temp_str += "C"
        elif temp_seq[j] == 8:
            temp_str += "G"
    train_seqs.append(temp_str)

for i in range(len(valid_file_x)):
    temp_seq= valid_file_x[i].numpy()
    temp_seq = temp_seq[1:-1]
    # convert 5 to a, 6 to u, 7 to c, 8 to g

    temp_str = ""
    for j in range(len(temp_seq)):
        if temp_seq[j] == 5:
            temp_str += "A"
        elif temp_seq[j] == 6:
            temp_str += "T"
        elif temp_seq[j] == 7:
            temp_str += "C"
        elif temp_seq[j] == 8:
            temp_str += "G"
    valid_seqs.append(temp_str)

for i in range(len(test_file_x)):
    temp_seq= test_file_x[i].numpy()
    temp_seq = temp_seq[1:-1]
    # convert 5 to a, 6 to u, 7 to c, 8 to g

    temp_str = ""
    for j in range(len(temp_seq)):
        if temp_seq[j] == 5:
            temp_str += "A"
        elif temp_seq[j] == 6:
            temp_str += "T"
        elif temp_seq[j] == 7:
            temp_str += "C"
        elif temp_seq[j] == 8:
            temp_str += "G"
    test_seqs.append(temp_str)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
config = transformers.BertConfig.from_pretrained(MODEL_NAME)
birnabert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,config=config,trust_remote_code=True)
birnabert.cls = torch.nn.Identity()

birnabert.to(device)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).float()
    labels = torch.stack(labels)
    return input_ids, attention_mask, labels


def get_embedding_vector(embedding):
    # take the max value of the embedding
    embedding = torch.max(embedding, dim=1)
    embedding = embedding.values

    return embedding.cpu().numpy()




class RNA_M6_Site(Dataset):
    def __init__(self, list, labels):
        self.list = list
        self.labels = labels
        
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx):
        seq = self.list[idx].replace('U', 'T')  # Replace 'U' with 'T'|
        seq = seq.upper()
        seq = dynamic_tokenize_preprocessing(seq)
        tokenized_output = tokenizer(seq, return_tensors="pt")
        tokenized_output = tokenized_output.to(device)
        input_ids = tokenized_output['input_ids'].squeeze(0)
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return  input_ids, label_tensor

train_dataset = RNA_M6_Site(train_seqs, label_train)
test_dataset = RNA_M6_Site(test_seqs, label_test)
val_dataset = RNA_M6_Site(valid_seqs, label_valid)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class finetuneSplicing(nn.Module):
    def __init__(self, model):
        super(finetuneSplicing, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids,attention_masks):
        output = self.model(input_ids, attention_mask = attention_masks).logits
        output = torch.mean(output, dim=1)
        output = self.fc1(output)
        output = self.sigmoid(output)
        return output.squeeze(1)
    

model = finetuneSplicing(birnabert)
model.to(device)

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=LR)
num_epochs = EPOCHS
num_steps = len(train_loader) * num_epochs
num_warmup_steps = WARMUP_RATIO * num_steps
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
loss_fn = nn.BCELoss()

# Metrics
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, recall_score

best_val_f1 = 0

# Train the model
model.train()
for epoch in range(num_epochs):
    epoch_losses_train = []
    all_predictions = []
    all_labels = []

    print(len(train_loader))
    for i, (input_ids, attention_masks, labels) in enumerate(tqdm(train_loader)):
        # print("Training step: ", i)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        labels = labels.float()
        optimizer.zero_grad()

        # Forward pass
        output = model(input_ids, attention_masks)
        # Compute loss
        loss = loss_fn(output, labels)
        epoch_losses_train.append(loss.item())

        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Convert outputs to binary (0 or 1) to match labels format
        binary_outputs = (output > 0.5).float()

        # Store predictions and labels for metrics calculation
        all_predictions.append(binary_outputs)
        all_labels.append(labels.int())

    avg_loss = sum(epoch_losses_train) / len(epoch_losses_train)
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    
    avg_accuracy = accuracy_score(all_labels, all_predictions)
    avg_f1_score = f1_score(all_labels, all_predictions)
    avg_recall = recall_score(all_labels, all_predictions)
    avg_mcc = matthews_corrcoef(all_labels, all_predictions)



    print(f"Train - Loss: {avg_loss:.4f}")


    model.eval()
    epoch_losses_test = []
    all_predictions = []
    all_labels = []

    for i, (input_ids, attention_masks, labels) in enumerate(tqdm(val_loader)):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        labels = labels.float()

        with torch.no_grad():
            output = model(input_ids, attention_masks)
            loss = loss_fn(output, labels)
            epoch_losses_test.append(loss.item())

            binary_outputs = (output > 0.5).float()

            all_predictions.append(binary_outputs)
            all_labels.append(labels.int())

    avg_loss = sum(epoch_losses_test) / len(epoch_losses_test)
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    
    avg_accuracy = accuracy_score(all_labels, all_predictions)
    avg_f1_score = f1_score(all_labels, all_predictions)
    avg_recall = recall_score(all_labels, all_predictions)
    avg_mcc = matthews_corrcoef(all_labels, all_predictions)

    print(f"Val Loss: {avg_loss:.4f}, ACC: {avg_accuracy:.4f}, F1: {avg_f1_score:.4f}, MCC: {avg_mcc:.4f}, R: {avg_recall:.4f}")

    if avg_f1_score > best_val_f1:
        best_val_f1 = avg_f1_score
        torch.save(model.state_dict(), "best_model.pth")


# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

model.eval()
epoch_losses_test = []
all_predictions = []
all_labels = []

for i, (input_ids, attention_masks, labels) in enumerate(tqdm(test_loader)):
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = labels.to(device)
    labels = labels.float()

    with torch.no_grad():
        output = model(input_ids, attention_masks)
        loss = loss_fn(output, labels)
        epoch_losses_test.append(loss.item())

        binary_outputs = (output > 0.5).float()

        all_predictions.append(binary_outputs)
        all_labels.append(labels.int())

avg_loss = sum(epoch_losses_test) / len(epoch_losses_test)
all_predictions = torch.cat(all_predictions).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()


avg_accuracy = accuracy_score(all_labels, all_predictions)
avg_f1_score = f1_score(all_labels, all_predictions)
avg_recall = recall_score(all_labels, all_predictions)
avg_mcc = matthews_corrcoef(all_labels, all_predictions)

print(f"Test Loss: {avg_loss:.4f}, ACC: {avg_accuracy:.4f}, F1: {avg_f1_score:.4f}, MCC: {avg_mcc:.4f}, R: {avg_recall:.4f}")
