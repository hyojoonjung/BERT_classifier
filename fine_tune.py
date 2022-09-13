import os
import torch
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn import metrics

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

k_fold = int(input("Enter the k-fold : "))
mode = input("Enter the Mode: fine_tune or train : ")
for i in range(k_fold):
## Parameters ##
    batch_size = 16
    fine_tune_epochs = 5
    train_epochs = 20
    language_model = 'roberta'
    base_model = '{}-base'.format(language_model)
    classification_model_path = 'model/fine_tune/{0}.pt'.format(base_model)
    fine_tune_model_path = "model/fine_tune/pandora/{0}/".format(base_model)

    ## Make Dir ##
    os.makedirs(fine_tune_model_path, exist_ok=True)

    ## Pretrained Model & Tokenizer ##
    if mode == 'fine_tune':
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=5, problem_type = "multi_label_classification") ##fine_tuning##
    elif mode == 'train':
        model = AutoModelForSequenceClassification.from_pretrained(fine_tune_model_path, num_labels=5, problem_type = "multi_label_classification") ##classfication##
    else:
        raise KeyError("Enter only fine_tune or train")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    ## Set up ##
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)


    ## Load dataframe ##
    # train_df = pd.read_csv('data/new_k_fold/train_{}.csv'.format(i+1), index_col=False)
    # valid_df = pd.read_csv('data/new_k_fold/valid_{}.csv'.format(i+1), index_col=False)
    # test_df = pd.read_csv('data/new_k_fold/test.csv', index_col=False)

    train_df = pd.read_csv('pandora/train_author.csv', index_col=False)
    valid_df = pd.read_csv('pandora/valid_author.csv', index_col=False)
    # test_df = pd.read_csv('data/k-fold_raw/test.csv', index_col=False)


    ## Make texts, labels ##
    def read_essay_split(df):
        texts = []
        labels = []
        for i in range(len(df)):
            label = []
            texts.append(df['TEXT'].iloc[i])

            # label.append(float(1) if df['OPN'].iloc[i] == 'y' else float(0))
            # label.append(float(1) if df['CON'].iloc[i] == 'y' else float(0))
            # label.append(float(1) if df['EXT'].iloc[i] == 'y' else float(0))
            # label.append(float(1) if df['AGR'].iloc[i] == 'y' else float(0))
            # label.append(float(1) if df['NEU'].iloc[i] == 'y' else float(0))

            label.append(float(df['OPN'].iloc[i]))
            label.append(float(df['CON'].iloc[i]))
            label.append(float(df['EXT'].iloc[i]))
            label.append(float(df['AGR'].iloc[i]))
            label.append(float(df['NEU'].iloc[i]))

            # label.append(float(df['cOPN'].iloc[i]))
            # label.append(float(df['cCON'].iloc[i]))
            # label.append(float(df['cEXT'].iloc[i]))
            # label.append(float(df['cAGR'].iloc[i]))
            # label.append(float(df['cNEU'].iloc[i]))

            labels.append(label)

        return texts, labels

    train_texts, train_labels = read_essay_split(train_df)
    val_texts, val_labels = read_essay_split(valid_df)
    # test_texts, test_labels = read_essay_split(test_df)

    ## tokenize text ##
    """
        Returns:
            input_ids : List of token ids to be fed to a model.
            token_type_ids = List of token type ids to be fed to a model, segment ids
            attention_mask : List of indices specifying which tokens should be attended to by the model
            etc. 
    """
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)


    ## CustomDataset ##
    class EssayDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            scaler = MinMaxScaler()
            self.labels = scaler.fit_transform(labels)
            # self.labels = labels
        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = EssayDataset(train_encodings, train_labels)
    val_dataset = EssayDataset(val_encodings, val_labels)
    # test_dataset = EssayDataset(test_encodings, test_labels)

    ## DataLoader ##
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


    def binary_label(label):
        rounded_label = torch.round(label)

        return rounded_label

    def pandora_label(labels):
        rounded_label = []
        for label in labels:
            b = []
            for sentiment in label:
                if sentiment > 0.5:
                    b.append(1.0)
                else:
                    b.append(0)
            rounded_label.append(b)

        return torch.tensor(rounded_label)

    ## Calculate accuracy ##
    def binary_accuracy(preds, y):
        
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        
        acc = torch.mean(correct, dim=0)

        return acc

    ## epoch_time ##
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    ## trainable parameter count ##
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## Trainer ##
    def train(train_loader):
        epoch_loss = 0
        epoch_acc_OPN = 0
        epoch_acc_CON = 0
        epoch_acc_EXT = 0
        epoch_acc_AGR = 0
        epoch_acc_NEU = 0
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            labels = pandora_label(labels).to(device)

            acc = binary_accuracy(outputs[1], labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc_OPN += acc[0].item()
            epoch_acc_CON += acc[1].item()
            epoch_acc_EXT += acc[2].item()
            epoch_acc_AGR += acc[3].item()
            epoch_acc_NEU += acc[4].item()

        return epoch_loss / len(train_loader), epoch_acc_OPN / len(train_loader), epoch_acc_CON / len(train_loader), epoch_acc_EXT / len(train_loader), epoch_acc_AGR / len(train_loader), epoch_acc_NEU / len(train_loader)

    ## Evaluate ##
    def evaluate(test_loader):
        epoch_loss = 0
        epoch_acc_OPN = 0
        epoch_acc_CON = 0
        epoch_acc_EXT = 0
        epoch_acc_AGR = 0
        epoch_acc_NEU = 0
        preds = []
        label_list = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                labels = pandora_label(labels).to(device)

                loss = outputs[0]
                acc = binary_accuracy(outputs[1], labels)
                epoch_loss += loss.item()
                epoch_acc_OPN += acc[0].item()
                epoch_acc_CON += acc[1].item()
                epoch_acc_EXT += acc[2].item()
                epoch_acc_AGR += acc[3].item()
                epoch_acc_NEU += acc[4].item()
                pred = torch.round(torch.sigmoid(outputs[1]))
                pred = pred.cpu().tolist()
                labels = labels.cpu().tolist()
                for i in range(len(pred)):
                    preds.append(pred[i])
                for i in range(len(labels)):
                    label_list.append(labels[i])
        label_names = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']

        preds = torch.tensor(preds).int()
        label_list = torch.tensor(label_list).int()

        # label_list = pd.DataFrame(label_list)
        # label_list.to_csv('label_list.csv', index=False)

        print(metrics.f1_score(label_list[:, 0].detach().numpy(), preds[:, 0].detach().numpy()))
        print(metrics.f1_score(label_list[:, 1].detach().numpy(), preds[:, 1].detach().numpy()))
        print(metrics.f1_score(label_list[:, 2].detach().numpy(), preds[:, 2].detach().numpy()))
        print(metrics.f1_score(label_list[:, 3].detach().numpy(), preds[:, 3].detach().numpy()))
        print(metrics.f1_score(label_list[:, 4].detach().numpy(), preds[:, 4].detach().numpy()))


        # F1_score_plot(label_list[:, 1].detach().numpy(), preds[:, 1].detach().numpy())
        # print(metrics.classification_report(label_list.detach().numpy(), preds.detach().numpy(), target_names=label_names))
        
        return epoch_loss / len(test_loader), epoch_acc_OPN / len(test_loader), epoch_acc_CON / len(test_loader), epoch_acc_EXT / len(test_loader), epoch_acc_AGR / len(test_loader), epoch_acc_NEU / len(test_loader)

    ## Epoch train ##
    def fine_tune(train_loader, val_loader, fine_tune_epochs):
        best_val_loss = float('inf')
        print("----Fine-tuning Start----")
        for epoch in range(fine_tune_epochs):
            start_time = time.time()
            train_loss, train_acc_OPN, train_acc_CON, train_acc_EXT, train_acc_AGR, train_acc_NEU = train(train_loader)
            val_loss, val_acc_OPN, val_acc_CON, val_acc_EXT, val_acc_AGR, val_acc_NEU = evaluate(val_loader)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1}/{fine_tune_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f"train loss : {train_loss:.3f} | train_acc_OPN : {train_acc_OPN*100:.4f}% | train_acc_CON : {train_acc_CON*100:.4f}% | train_acc_EXT : {train_acc_EXT*100:.4f}% | train_acc_AGR : {train_acc_AGR*100:.4f}% | train_acc_NEU : {train_acc_NEU*100:.4f}%")
            print(f"val loss : {val_loss:.3f} | val_acc_OPN : {val_acc_OPN*100:.4f}% | val_acc_CON : {val_acc_CON*100:.4f}% | val_acc_EXT : {val_acc_EXT*100:.4f}% | val_acc_AGR : {val_acc_AGR*100:.4f}% | val_acc_NEU : {val_acc_NEU*100:.4f}%")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("----Saving fine tuned Model----")
                torch.save(model.state_dict(), fine_tune_model_path+'model_{}.pt'.format(i))
                print("----Model Saved!----")

        print("----Fine-tuning Complete----")

    def classification_train(train_loader, val_loader, train_epochs):
        best_val_loss = float('inf')
        # for name, param in model.named_parameters():                
        #     if name.startswith(language_model):
        #         param.requires_grad = False

        # for name, param in model.named_parameters():                
        #     if param.requires_grad:
        #         print(name)

        print(f'The model has {count_parameters(model):,} trainable parameters')
        
        print("----Training Start----")
        for epoch in range(train_epochs):
            start_time = time.time()
            train_loss, train_acc_OPN, train_acc_CON, train_acc_EXT, train_acc_AGR, train_acc_NEU = train(train_loader)
            val_loss, val_acc_OPN, val_acc_CON, val_acc_EXT, val_acc_AGR, val_acc_NEU = evaluate(val_loader)
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1}/{train_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f"train loss : {train_loss:.3f} | train_acc_OPN : {train_acc_OPN*100:.4f}% | train_acc_CON : {train_acc_CON*100:.4f}% | train_acc_EXT : {train_acc_EXT*100:.4f}% | train_acc_AGR : {train_acc_AGR*100:.4f}% | train_acc_NEU : {train_acc_NEU*100:.4f}%")
            print(f"val loss : {val_loss:.3f} | val_acc_OPN : {val_acc_OPN*100:.4f}% | val_acc_CON : {val_acc_CON*100:.4f}% | val_acc_EXT : {val_acc_EXT*100:.4f}% | val_acc_AGR : {val_acc_AGR*100:.4f}% | val_acc_NEU : {val_acc_NEU*100:.4f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("----Saving Classification Model----")
                torch.save(model.state_dict(), classification_model_path)
                print("----Model Saved!----")

        print("----Training Complete----")
    def F1_score_plot(labels, preds):
        import numpy as np
        from sklearn.preprocessing import Binarizer
        thresholds = np.arange(0.1, 1, 0.1)
        preds = np.reshape(preds, (-1, 1))
        for custom_threshold in thresholds:
            binarizer = Binarizer(threshold=custom_threshold).fit(preds)
            custom_predict = binarizer.transform(preds)
            print("threshold: ", custom_threshold)
            print("-" * 60)
            metrics.f1_score(labels, custom_predict, average='macro')
            print("=" * 60) 

    def test(test_loader):
        model.load_state_dict(torch.load(fine_tune_model_path+'model_{}.pt'.format(i)))
        print("----Testing Start----")
        test_loss, test_acc_OPN, test_acc_CON, test_acc_EXT, test_acc_AGR, test_acc_NEU = evaluate(test_loader)
        total = (test_acc_OPN + test_acc_CON + test_acc_EXT + test_acc_AGR + test_acc_NEU) / 5
        print(f"test loss : {test_loss:.3f} | test_acc_OPN : {test_acc_OPN*100:.4f}% | test_acc_CON : {test_acc_CON*100:.4f}% | test_acc_EXT : {test_acc_EXT*100:.4f}% | test_acc_AGR : {test_acc_AGR*100:.4f}% | test_acc_NEU : {test_acc_NEU*100:.4f}%")
        print(f"Average : {total*100:.4f}%")
        print("----Testing Compelte----")

    if mode == 'fine_tune':
        fine_tune(train_loader, val_loader, fine_tune_epochs)
        # test(test_loader)
    elif mode == 'train':
        classification_train(train_loader, val_loader, train_epochs)
        test(test_loader)

    del model, tokenizer