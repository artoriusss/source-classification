# Classification of media sources

## Installation 
Clone the repo:
```
https://github.com/artoriusss/source-classification.git
```
Create anaconda environment
```
conda create --name src_cls python=3.10
conda activate src_cls
```

Download the raw dataset
```
kaggle competitions download -c ai-defence-summer-school-2024
```
If the above command fails, please ensure a valid `kaggle.json` with your credentials is present in `~/.kaggle/` directory

```
mkdir -p data/raw data/preprocessed
```
```
unzip ai-defence-summer-school-2024.zip -d data/raw && rm ai-defence-summer-school-2024.zip
```

Finally, install the required dependencies

```
pip install -r requirements.txt
```

## Preprocessing
Within this project you can choose two preprocessing methods: light/soft or hard. Hard preprocessing is usually a good practise for classical ML methods, since it lowercases the text and removes more elements that are not easily handled by these models, such as: emojis, hashtags, mentions, urls, etc. Light preprocessing is usually applied to larger models, such a BERT, RoBERTa, etc.

To run soft preprocessing, type the commment.
```
python -m Preprocessor.script --light
```
For hard preprocessing, use a flag `--hard`. 

## Models
We trained and experimented with a number of different models. Classical ML models: Random Forests classifier. Large Language Models: fine-tuned XLM-Roberta-base. All of the pickled classical models are accessible from ```/models``` directory. 

Fine-tuned Roberta can be download with the following script:

```
import kagglehub

# Download the latest version of the model to the models/ directory
path = kagglehub.model_download("artoriusss/roberta-text-classify/pyTorch/roberta", path='models')

print("Path to model files:", path)
```