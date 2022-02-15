from dont_patronize_me import DontPatronizeMe
import logger as logging
from logger import logger
import pandas as pd

def load_task1():
    '''
    return a pandas dataframe with paragraphs and labels
    '''
    dpm = DontPatronizeMe('data', 'dontpatronizeme_pcl.tsv')
    dpm.load_task1()
    logger.info(f"task1 loaded ({dpm.train_task1_df.shape[0]} rows, {dpm.train_task1_df.shape[1]} columns)")
    return dpm.train_task1_df

def load_paragraph_ids():
    train_ids = pd.read_csv('data/train_semeval_parids-labels.csv')
    test_ids = pd.read_csv('data/dev_semeval_parids-labels.csv')
    train_ids.par_id = train_ids.par_id.astype(str)
    test_ids.par_id = test_ids.par_id.astype(str)
    logger.info(f"training data size: {len(train_ids)}; test data size: {len(test_ids)}")
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
        for _, row in df.iterrows():
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
    train_data = rebuild_raw_dataset(dpm, train_ids)
    test_data = rebuild_raw_dataset(dpm, test_ids)
    
    save_raw_data(train_data, 'data/train.tsv')
    save_raw_data(test_data, 'data/test.tsv')
    
    # train_data, test_data = load_data()
    
    # texts = train_data['text'].astype(str).values.tolist()
    # save_raw_sentences(texts, 'data/raw_sentences_train.txt')
    # labels = train_data['label'].astype(str).values.tolist()
    # save_sentences_and_labels(texts, labels, "data/sentences_and_labels_train.txt")
    
    # texts = test_data['text'].astype(str).values.tolist()
    # save_raw_sentences(texts, 'data/raw_sentences_test.txt')
    # labels = test_data['label'].astype(str).values.tolist()
    # save_sentences_and_labels(texts, labels, "data/sentences_and_labels_test.txt")