from pathlib import Path

FASTTEXT_PATH = Path("utils/models/lid.176.bin")
STOPWORDS_PATH = Path("utils/stopwords.txt")

DATA_PATH = Path("data/")
RAW_PATH = DATA_PATH / "raw/"
PREPROCESSED_PATH = DATA_PATH / "preprocessed/"

MESSAGES_PATH = RAW_PATH / "messages" / "messages.jsonl"
TRAIN_PATH = PREPROCESSED_PATH / "train.csv"
TEST_PATH = PREPROCESSED_PATH / "test.csv"