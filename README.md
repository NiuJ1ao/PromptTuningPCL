# Prompt Tuning in Condescending Detection

## Requirements
- [Pytorch](https://pytorch.org/get-started/locally/) please select the correct CUDA version matched your GPU.
- other requirements
```
pip install -r requirements.txt
```
If some packages are still missing, please use ```pip install <package_name>```

## Baseline
```
python baseline.py
```

## Train
If you would like to change hyperparameters or models, please edit the file directly (too lazy to write CLI)
```
python train.py
```
## Evaluate
```
python evaluate.py
```
## Predict
```
python predict.py
```
