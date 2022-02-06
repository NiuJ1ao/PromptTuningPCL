from dont_patronize_me import DontPatronizeMe
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
    
def rebuild_data_set(dpm, ids):
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

def load_data():
    dpm = load_task1()
    train_ids, test_ids = load_paragraph_ids()
    return rebuild_data_set(dpm, train_ids), rebuild_data_set(dpm, test_ids)
    