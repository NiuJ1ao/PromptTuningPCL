from dont_patronize_me import DontPatronizeMe
import torch
import logger as logging
from logger import logger
import pandas as pd
import os

class PCLDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset="train", is_augment=True, max_length=128):
        # get data
        if dataset == "train":
            df, _ = load_data()
            
            if is_augment:
                dpm = DontPatronizeMe('data', '')
                dpm.load_task1(file_name="augment_positive_paraphrase.tsv")
                augments = dpm.train_task1_df
                logger.info(f"Augmented positive data size: {len(augments)}")
                df = pd.concat([df, augments])
                # dpm.load_task1(file_name="augment_positive_EDA.tsv")
                # augments = dpm.train_task1_df
                # logger.info(f"Augmented positive data size: {len(augments)}")
                # df = pd.concat([df, augments])
            else:
                # Downsampling
                pcldf = df[df.label==1]
                npos = len(pcldf)
                df = pd.concat([pcldf, df[df.label==0][:npos*2]])
        elif dataset == "test":
            # _, df = load_data()
            dpm = DontPatronizeMe('data', '')
            dpm.load_task1(file_name="test.tsv")
            df = dpm.train_task1_df
            
            # # analysis question
            # df = df[df.orig_label=="2"]
            # assert(len(df)>0), len(df)
        else:
            assert False
            
        logger.info(f"Dataset {dataset}: Positive data size = {len(df[df.label==1])}; Negetive data size = {len(df[df.label==0])}")
        self.tokenizer = tokenizer
        self.snts = df.text.astype(str).values.tolist()
        self.labels = df.label.astype(int).values.tolist()
        self.encodings = self.tokenizer(self.snts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
class PCLTestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_length=128):
        dpm = DontPatronizeMe('data', 'data/task4_test.tsv')
        dpm.load_test()
        df = dpm.test_set_df
        
        self.snts = df.text.astype(str).values.tolist()
        self.encodings = tokenizer(self.snts, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
        
    def __len__(self):
        return len(self.snts)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

def load_task1():
    '''
    return a pandas dataframe with paragraphs and labels
    '''
    dpm = DontPatronizeMe('data', '')
    dpm.load_task1()
    logger.info(f"Task1 loaded ({dpm.train_task1_df.shape[0]} rows, {dpm.train_task1_df.shape[1]} columns)")
    return dpm.train_task1_df

def load_paragraph_ids(folder="data"):
    path = os.path.join(folder, "paragraph_ids.csv")
    train_ids = pd.read_csv(os.path.join(folder, "train_semeval_parids-labels.csv"))
    test_ids = pd.read_csv(os.path.join(folder, "dev_semeval_parids-labels.csv"))
    train_ids.par_id = train_ids.par_id.astype(str)
    test_ids.par_id = test_ids.par_id.astype(str)
    logger.info(f"Training data size: {len(train_ids)}; test data size: {len(test_ids)}")
    return train_ids, test_ids
    
def rebuild_dataset(dpm, ids):
    rows = [] # will contain par_id, label and text
    for idx in range(len(ids)):  
        parid = ids.par_id[idx]

        # select row from original dataset
        text = dpm.loc[dpm.par_id == parid].text.values[0]
        label = dpm.loc[dpm.par_id == parid].label.values[0]
        rows.append({
            'par_id':parid,
            'text':text,
            'label':label
        })
    return pd.DataFrame(rows)

def rebuild_raw_dataset(dpm, ids):
    rows = [] # will contain par_id, label and text
    for idx in range(len(ids)):  
        parid = ids.par_id[idx]

        # select row from original dataset
        text = dpm.loc[dpm.par_id == parid].text.values[0]
        label = dpm.loc[dpm.par_id == parid].label.values[0]
        orig_label = dpm.loc[dpm.par_id == parid].orig_label.values[0]
        art_id = dpm.loc[dpm.par_id == parid].art_id.values[0]
        keyword = dpm.loc[dpm.par_id == parid].keyword.values[0]
        country = dpm.loc[dpm.par_id == parid].country.values[0]
        rows.append({
            'par_id':parid,
            'text':text,
            'label':label,
            'orig_label': orig_label,
            'art_id': art_id,
            'keyword': keyword,
            'country': country
        })
    return pd.DataFrame(rows)

def load_data():
    dpm = load_task1()
    train_ids, test_ids = load_paragraph_ids()
    return rebuild_dataset(dpm, train_ids), rebuild_dataset(dpm, test_ids)

def save_sentences_and_labels(sentences, labels, filename):
    with open(filename, 'w') as f:
        logger.info(f"Saving {len(sentences)} pairs to {filename}")
        for sentence, label in zip(sentences, labels):
            f.write(f'{sentence}\t{label}\n')
            
def save_raw_data(df, filename):
    df = df.reset_index()
    with open(filename, 'w') as f:
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        for _, row in df.iterrows():
            if len(row.text.split()) > 63:
                f.write(f"{row.par_id}\t{row.art_id}\t{row.keyword}\t{row.country}\t{row.text}\t{row.orig_label}\n")
            
def save_raw_sentences(sentences, filename):
    with open(filename, 'w') as f:
        logger.info(f"Saving {len(sentences)} sentences to {filename}")
        for snt in sentences:
            f.write(f'{snt}\n')
            
if __name__ == "__main__":
    logging.init_logger()
    
    dpm = load_task1()
    train_ids, test_ids = load_paragraph_ids()
    # train_data = rebuild_raw_dataset(dpm, train_ids)
    test_data = rebuild_raw_dataset(dpm, test_ids)
    
    # save_raw_data(train_data, 'data/train.tsv')
    save_raw_data(test_data, 'data/test_long.tsv')
    
    # train_data, test_data = load_data()
    
    # texts = train_data['text'].astype(str).values.tolist()
    # save_raw_sentences(texts, 'data/raw_sentences_train.txt')
    # labels = train_data['label'].astype(str).values.tolist()
    # save_sentences_and_labels(texts, labels, "data/sentences_and_labels_train.txt")
    
    # texts = test_data['text'].astype(str).values.tolist()
    # save_raw_sentences(texts, 'data/raw_sentences_test.txt')
    # labels = test_data['label'].astype(str).values.tolist()
    # save_sentences_and_labels(texts, labels, "data/sentences_and_labels_test.txt")