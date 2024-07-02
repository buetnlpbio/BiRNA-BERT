BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 20
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 1
MODEL_NAME = "../../MODEL"
TOKENIZER = "../../TOKENIZER"


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import transformers
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup, AutoModelForMaskedLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
config = transformers.BertConfig.from_pretrained(MODEL_NAME)
birnabert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,config=config,trust_remote_code=True)
birnabert.cls = torch.nn.Identity()

birnabert.to(device)




def collate_fn(batch):
    # Separate sequences and angle matrices
    input_ids, angle_matrices = zip(*batch)

    # Pad sequences (input_ids)
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)  # Assuming 0 is the padding index for 

    attention_masks = (padded_input_ids != tokenizer.pad_token_id).float()
    padded_angle_matrices = pad_sequence(angle_matrices, batch_first=True, padding_value=float('nan'))  # Use nan for padding so it can be easily ignored in loss calculations

    return padded_input_ids, attention_masks, padded_angle_matrices


MAX_LENGTH = 200
class RNADataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with RNA data.
        """
        # Load the data
        df = pd.read_csv(csv_file)
        df.sort_values(by='id', inplace=True)

        # Filter out rows with bases not in ['A', 'U', 'C', 'G']
        df = df[df['Base'].isin(['A', 'U', 'C', 'G'])]

        # Replace '---' with np.nan and convert to numeric, handling errors
        angle_columns = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
        df.replace('---', np.nan, inplace=True)
        for col in angle_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts errors to NaN

        # Optionally, fill NaN values with a placeholder (with mean of the column)
        df.fillna(0, inplace=True)  # filling with mean instead of 0 for better data handling

        # Convert angles from degrees to radians since trig functions use radians
        df[angle_columns] = np.deg2rad(df[angle_columns])

        # Group by 'id' and store sequences and angle matrices
        self.sequences = []
        self.angle_matrices = []
        self.lengths= []

        for _, group in df.groupby('id'):
            self.sequences.append(group['Base'].tolist())
            self.lengths.append(len(self.sequences[-1]))
            # Create sine and cosine for each angle
            angles_sincos = []
            for _, row in group.iterrows():
                angle_row_sincos = []
                for angle in angle_columns:
                    angle_row_sincos.extend([np.sin(row[angle]), np.cos(row[angle])])
                angles_sincos.append(angle_row_sincos)
            self.angle_matrices.append(angles_sincos)
            if self.lengths[-1] > MAX_LENGTH:
                self.lengths[-1] = MAX_LENGTH
                self.sequences[-1] = self.sequences[-1][:MAX_LENGTH]
                self.angle_matrices[-1] = self.angle_matrices[-1][:MAX_LENGTH]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = ''.join(seq)
        seq = seq.upper()
        seq = ' '.join(seq) #nucleotide tokenization
        seq = seq.strip()
        seq = seq.replace("U", "T")

        tokenized_output = tokenizer(seq, return_tensors="pt")
        tokenized_output = tokenized_output.to(device)

        
        # remove the first and last token
        # tokenized_output['input_ids'] = tokenized_output['input_ids'][0][1:-1]
        input_ids = tokenized_output['input_ids'].squeeze(0)

        angles = self.angle_matrices[idx]

        # Convert angles to a torch tensor
        angles_tensor = torch.tensor(angles, dtype=torch.float)  # This converts the expanded list into a 2D tensor directly

       
        return input_ids, angles_tensor
    def get_max_min_lengths(self):
        if not self.lengths:
            return None, None
        return max(self.lengths), min(self.lengths)

# Usage
dataset_train = RNADataset('data/fixed_final_output_TR_seqs.csv')
validation_dataset = RNADataset('data/fixed_final_output_VL_seqs.csv')
test1_dataset = RNADataset('data/fixed_final_output_TS1_seqs.csv')
test2_dataset = RNADataset('data/fixed_final_output_TS2_seqs.csv')
test3_dataset = RNADataset('data/fixed_final_output_TS3_seqs.csv')



# Print max and min lengths
max_len, min_len = dataset_train.get_max_min_lengths()
print(f"Training Dataset - Max Length: {max_len}, Min Length: {min_len}")

max_len, min_len = validation_dataset.get_max_min_lengths()
print(f"Validation Dataset - Max Length: {max_len}, Min Length: {min_len}")

max_len, min_len = test1_dataset.get_max_min_lengths()
print(f"Test1 Dataset - Max Length: {max_len}, Min Length: {min_len}")

max_len, min_len = test2_dataset.get_max_min_lengths()
print(f"Test2 Dataset - Max Length: {max_len}, Min Length: {min_len}")

max_len, min_len = test3_dataset.get_max_min_lengths()
print(f"Test3 Dataset - Max Length: {max_len}, Min Length: {min_len}")





class FineTuneAngle(nn.Module):
    def __init__(self, birnabert):
        super(FineTuneAngle, self).__init__()
        self.model = birnabert  # Assuming birnabert is a pretrained BERT-like model
        self.fc1 = nn.Linear(768, 256)  # Typically, 768 is the dimension of BERT output features
        self.fc2 = nn.Linear(256, 14)   # Assuming there are 14 regression targets
        
        self.tanh = nn.Tanh()
        
    def forward(self, input_ids, attention_masks):
        output = self.model(input_ids, attention_mask=attention_masks)
        output = output.logits
        output = output[:, 1:-1, :]  # Assuming you want to trim outputs from both ends for some reason
        
        output = self.fc1(output)
        output = torch.relu(output)  # Applying ReLU activation function
        output = self.fc2(output)
        output = self.tanh(output)  # Ensuring output is between -1 and 1
        return output




def custom_loss(output, target):
    # Assume target could be 2D: [batch_size, num_features]
    # Create a mask where neither element of the feature pair is NaN (assuming outputs are pairs)
    mask = ~torch.isnan(target).any(dim=2)  # Check along the feature dimension
    # print(mask)

    # Filter out the batches where the target has NaNs
    # The mask will be broadcasted to match the dimensions
    if mask.any():
        valid_output = output[mask]
        valid_target = target[mask]
        loss = F.mse_loss(valid_output, valid_target, reduction='mean')
    else:
        print("HELLO")
        # If all are NaNs, give a dummy loss (this should be handled to avoid)
        loss = torch.tensor(0.0, device=output.device)

    return loss

# Placeholder for your DataLoader instances

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_data_loader_1 = DataLoader(test1_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_data_loader_2 = DataLoader(test2_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_data_loader_3 = DataLoader(test3_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)



# Assuming birnabert is already defined somewhere and accessible
model = FineTuneAngle(birnabert)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
num_steps = len(train_dataloader) * EPOCHS
num_warmpup_steps = WARMUP_RATIO * num_steps
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmpup_steps, last_epoch=-1)
# optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()  # Using Mean Squared Error loss for training


def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    total_loss = 0
    print("Training")
    print(len(dataloader))
    for batch, (input_ids, attention_masks, labels) in enumerate(dataloader):
        # print(batch)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        # Forward pass

        optimizer.zero_grad()
        
        predictions = model(input_ids, attention_masks)

        loss = custom_loss(predictions, labels)
        
        # Backward and optimize

        if (batch+1) % GRAD_ACCUM_STEPS == 0:
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Training loss: {average_loss:.2f}")



def test_loop(dataloader, model, name, device='cuda'):
    model.eval()
    total_maes = [0] * 7  # Initialize a list to store the cumulative MAE for each angle
    count_samples = [0] * 7  # This will count the non-NaN samples for each angle to average correctly

    with torch.no_grad():
        for batch, (input_ids,attention_masks,labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            predictions = model(input_ids, attention_masks)

            # Calculate angles and MAE for each angle pair
            for i in range(7):  # Assuming there are 7 angles, hence 14 outputs (sine, cosine pairs)
                pred_angle = torch.atan2(predictions[:, 2*i], predictions[:, 2*i+1]) * (180 / np.pi)
                true_angle = torch.atan2(labels[:, 2*i], labels[:, 2*i+1]) * (180 / np.pi)

                # Mask to ignore NaNs
                valid_mask = ~torch.isnan(true_angle)

                # Only compute MAE on valid data (non-NaN)
                if valid_mask.any():  # Check if there's any valid data
                    abs_diff = torch.abs(pred_angle[valid_mask] - true_angle[valid_mask])
                    adjusted_diff = torch.minimum(abs_diff, 360 - abs_diff)
                    mae = adjusted_diff.mean().item()

                    total_maes[i] += mae
                    count_samples[i] += valid_mask.sum().item()  # Sum up valid samples

        # Compute the average MAE for each angle across all valid entries
        average_maes = [total_maes[i] / len(dataloader) for i in range(7)]
        # for i, mae in enumerate(average_maes):
        #     print(f"Mean Absolute Error for angle {i+1}: {mae:.2f} degrees")
        print(name, sum(average_maes), average_maes)


num_epochs = EPOCHS
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n----------------------------------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer, scheduler)
    test_loop(validation_dataloader, model, "V ")
    test_loop(test_data_loader_1, model, "T1")
    test_loop(test_data_loader_2, model, "T2")
    test_loop(test_data_loader_3, model, "T3")
    print()

print("Done!")