import sys
sys.path.append('..')

import random
from data_loader import *

verbalizer = ["No", "Yes"]
pcl_tokens = ["patronizing", "condescending"]
MASK = '<mask>'
SEP = '</s>'

# df = load_task1()
# train_ids, test_ids = load_paragraph_ids()
# train_df = rebuild_raw_dataset(df, train_ids)

dpm = DontPatronizeMe('data', 'data/task4_test.tsv')
# dpm.load_task1(file_name="augment_positive.tsv")
# augments = dpm.train_task1_df
# logger.info(f"Augmented positive data size: {len(augments)}")
# train_df = pd.concat([train_df, augments])

            
def generate_prompts(file, df):
    df = df.reset_index()
    for _, line in df.iterrows():
        try:
            pcl_token = random.choice(pcl_tokens)
            label = int(line.label)
            target = f"{line.text} {SEP} {verbalizer[label]} , this paragraph is {pcl_token} for {line.keyword} ."
            prompt = f"{line.text} {SEP} {MASK} , this paragraph is {pcl_token} for {line.keyword} .\t{target}"
            file.write(f"{prompt}\n")
        except Exception as e:
            print(e)
            print(line)

# with open(f"data/prompts_train.txt", "w") as f:
#     generate_prompts(f, train_df)
            

# test_df = rebuild_raw_dataset(df, test_ids)
# with open(f"data/prompts_test.txt", "w") as f:
#     generate_prompts(f, test_df)

dpm.load_test()
with open("data/task4_test_prompt.txt", "w") as f:
    df = dpm.test_set_df.reset_index()
    for _, line in df.iterrows():
        try:
            pcl_token = random.choice(pcl_tokens)
            prompt = f"{line.text} {SEP} {MASK} , this paragraph is not {pcl_token} for {line.keyword} ."
            f.write(f"{prompt}\n")
        except Exception as e:
            print(e)
            print(line)