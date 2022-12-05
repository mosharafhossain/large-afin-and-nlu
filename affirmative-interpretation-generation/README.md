
## Requirements
Python 3.7+   
List of the python packages are provided in ./env-setup/requirements.txt file.  

- Install required packages
```bash
pip install --upgrade pip
pip install -r ./env-setup/requirements.txt
```

- Download T5-Large transformer:
```bash
cd your_directory/affirmative-interpretation-generation
cd ./model/pre-trained/T5-large
wget https://huggingface.co/t5-large/resolve/main/config.json
wget https://huggingface.co/t5-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/t5-large/resolve/main/spiece.model
wget https://huggingface.co/t5-large/resolve/main/tokenizer.json
cd ../../../
```

## Datasets
We have created a dataset of 153,273 pairs of sentences containing negation and their affirmative interpretations. [Click here](https://github.com/mosharafhossain/large-afin-and-nlu/blob/main/affirmative-interpretation-generation/data/large-afin/large-afin.jsonl) to download the dataset. As part of the training and evaluation purposes, we used AFIN, a small corpus of negated sentences and their affirmative interpretations. AFIN can be downloaded from [here](https://github.com/mosharafhossain/AFIN/tree/main/data/splits).



## How to Run

- Train the affirmative interpretation generator (config.json provides required hyperparameters)
```bash
  python train.py -c ./config/config.json
```
- Generate affirmative interpretations of the negations in the dataset of AFIN
```bash
  python generate.py -c ./config/config.json
```
- Generate affirmative interpretations of negations from a user specified (text) file. The file must contain one sentence per line. 
```bash
  python generate.py -c ./config/config.json -inp USER_INPUT_PATH -out USER_OUTPUT_PATH
```


## How to Evaluate 
- Evaluate the trained system with three automatic metrics (i.e., BLUE-2, chrf++, and METEOR) on the test dataset of AFIN):
```bash
python scorer.py -g ./data/afin/test.jsonl -s ./output/test/output_test.jsonl

```
