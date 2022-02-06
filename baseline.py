import logger as logging
from logger import logger
from data_loader import load_data
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
from collections import Counter
import util
from sklearn.metrics import f1_score

def baseline():
    logging.init_logger(log_level=logging.INFO)
    train_df, test_df = load_data()
    
    pcldf = train_df[train_df.label==1]
    npos = len(pcldf)
    logger.info(f"positive samples = {npos}; negetive samples = {len(train_df) - npos}")

    downsampled_training_set = pd.concat([pcldf,train_df[train_df.label==0][:npos*2]])
    pcldf = downsampled_training_set[downsampled_training_set.label==1]
    logger.info(f"After downsampling: positive samples =  {len(pcldf)}; negetive samples = {len(downsampled_training_set) - len(pcldf)}")
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"Cuda available? {cuda_available}")
    
    task1_model_args = ClassificationArgs(num_train_epochs=1, 
                                      no_save=True, 
                                      no_cache=True, 
                                      overwrite_output_dir=True)
    task1_model = ClassificationModel("roberta", 
                                      'roberta-base', 
                                      args = task1_model_args, 
                                      num_labels=2, 
                                      use_cuda=cuda_available)
    # train model
    task1_model.train_model(downsampled_training_set[['text', 'label']])

    # run predictions
    preds_task1, _ = task1_model.predict(test_df.text.tolist())
    logger.info(f"Prediction distribution: {Counter(preds_task1)}")
    
    # evaluate
    f1_score = evaluate(preds_task1, test_df.label.tolist())
    logger.info(f"F1 score: {f1_score}")
    
    util.labels2file([[k] for k in preds_task1], f"predictions/baseline_{f1_score}.txt")
    
def baseline_biased():
    logging.init_logger(log_level=logging.INFO)
    train_df, test_df = load_data()
    
    pcldf = train_df[train_df.label==1]
    npos = len(pcldf)
    logger.info(f"positive samples = {npos}; negetive samples = {len(train_df) - npos}")

    # downsampled_training_set = pd.concat([pcldf,train_df[train_df.label==0][:npos*2]])
    # pcldf = downsampled_training_set[downsampled_training_set.label==1]
    # logger.info(f"After downsampling: positive samples =  {len(pcldf)}; negetive samples = {len(downsampled_training_set) - len(pcldf)}")
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"Cuda available? {cuda_available}")
    
    task1_model_args = ClassificationArgs(num_train_epochs=1, 
                                      no_save=True, 
                                      no_cache=True, 
                                      overwrite_output_dir=True)
    task1_model = ClassificationModel("roberta", 
                                      'roberta-base', 
                                      args = task1_model_args, 
                                      num_labels=2, 
                                      use_cuda=cuda_available)
    # train model
    task1_model.train_model(train_df[['text', 'label']])

    # run predictions
    preds_task1, _ = task1_model.predict(test_df.text.tolist())
    logger.info(f"Prediction distribution: {Counter(preds_task1)}")
    
    # evaluate
    f1_score = evaluate(preds_task1, test_df.label.tolist())
    logger.info(f"F1 score: {f1_score}")
    
    util.labels2file([[k] for k in preds_task1], f"predictions/baseline_biased_{f1_score}.txt")
    
def evaluate(preds, golds):
    return f1_score(golds, preds, average='binary')
    
def main():
    baseline()
    baseline_biased()
    
if __name__ == "__main__":
    main()
    