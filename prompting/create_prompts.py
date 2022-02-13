import random

verbalizer = ["No", "Yes"]
pcl_tokens = ["patronizing", "condescending"]
mask_token = "<MASK>" # roberta

for data in ["train", "test"]:
    with open(f"data/sentences_and_labels_{data}.txt", "r") as fin:
        with open(f"data/prompts_{data}.txt", "w") as fout:
            for line in fin:
                try:
                    line_segs = line.strip().split('\t')
                    if len(line_segs) < 2: 
                        paragraph = ""
                        label = line_segs[0]
                    else:
                        paragraph, label = line_segs[0], line_segs[1]
                    pcl_token = random.choice(pcl_tokens)
                    prompt = f"{paragraph} Is this paragraph {pcl_token} ? {mask_token} .\t{verbalizer[int(label)]}"
                    fout.write(f"{prompt}\n")
                    fout.write(f"{prompt}\n")
                except Exception as e:
                    print(e)
                    print(line)
                    print(line_segs)