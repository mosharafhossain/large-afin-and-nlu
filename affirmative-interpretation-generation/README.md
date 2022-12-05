
## Requirements
Python 3.7
List of the python packages are provided in ./env-setup/requirements.txt file.  

```bash
# Install required packages
pip install --upgrade pip
pip install -r ./env-setup/requirements.txt

Download T5-Large:
cd your_directory/affirmative-interpretation-generation
cd ./model/pre-trained/T5-large
wget https://huggingface.co/t5-large/resolve/main/config.json
wget https://huggingface.co/t5-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/t5-large/resolve/main/spiece.model
wget https://huggingface.co/t5-large/resolve/main/tokenizer.json
cd ../../../
```

## How to Run

```bash
  # Train the affirmative interpretation generator
  python train.py -c ./config/config.json
  
  # Generate affirmative interpretation of negation in the AFIN test dataset
  python generate.py -c ./config/config.json
  
  # Generate affirmative interpretation of negation of a user specified (text) file. The input file must contain one sentence per line.
  python generate.py -c ./config/config.json -inp USER_INPUT_PATH -out USER_OUTPUT_PATH
```


## How to Evaluate (with test dataset of AFIN)
```bash
python scorer.py -g ./data/afin/test.jsonl -s ./output/test/output_test.jsonl

```