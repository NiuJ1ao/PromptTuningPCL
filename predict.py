import numpy as np
from logger import logger
import logger as logging
from collections import Counter
from data_loader import PCLTestDataset
from transformers import Trainer
from transformers import BartTokenizer, BartForSequenceClassification

logging.init_logger()

best_model = "bart_large_paraphrases_2/checkpoint-1308"

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForSequenceClassification.from_pretrained(best_model, num_labels=2)

test_data = PCLTestDataset(tokenizer, max_length=128)

trainer = Trainer(model=model)

outputs = trainer.predict(test_data)
logits, _ = outputs.predictions

preds = np.argmax(logits, axis=-1)
logger.info(f"Prediction distribution: {Counter(preds)}")

with open(f"task1.txt", 'w') as f:
    for pi in preds:
        f.write(f'{pi}\n')