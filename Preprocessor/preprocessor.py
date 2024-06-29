from pathlib import Path
import os
import re
import ast
import emoji
import fasttext
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect
from tqdm import tqdm
from nltk import tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.model_selection import train_test_split

from .paths import *


from langdetect import detect, LangDetectException

class Preprocessor:
    def __init__(self, model_path, stopwords_path):
        print(f"Loading model from: {str(model_path)}")
        print(f"Loading stopwords from: {str(stopwords_path)}")
        self.stopwords_all = self.read_stopwords(str(stopwords_path))
        self.emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

    def read_stopwords(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read().strip()
            preprocessed_file_content = ast.literal_eval(file_content)
            return set(preprocessed_file_content)

    @staticmethod
    def collapse_dots(input):
        input = re.sub("\.+", ".", input)
        all_collapsed = False
        while not all_collapsed:
            output = re.sub(r"\.(( )*)\.", ".", input)
            all_collapsed = input == output
            input = output
        return output

    def preprocess_hard(self, input):
        words_list = []
        result_str = ''
        for token in word_tokenize(input):
            if token.lower() not in self.stopwords_all:
                words_list.append(token)
            result_str = " ".join(words_list)
        return result_str

    def process(self, input, light=False):
        if not isinstance(input, str):
            return input

        input = " ".join(tokenize.sent_tokenize(input))

        input = re.sub(r"http\S+", "", input)
        input = re.sub(r"\n+", ". ", input)
        input = re.sub(self.emoji_pattern, '', input)
        input = re.sub(r"\bt\.me/\S+", "", input)

        for symb in ["!", ",", ":", ";", "?", "_"]:
            input = re.sub(rf"\{symb}\.", symb, input)

        input = re.sub(r"#\S+", "", input)
        input = re.sub(r"@\S+", "", input)
        input = self.collapse_dots(input)
        input = input.strip()

        if light:
            return input

        return self.preprocess_hard(input)

    @staticmethod
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def remove_between_square_brackets(text):
        return re.sub("\[[^]]*\]", "", text)

    @staticmethod
    def remove_urls(text):
        return re.sub(r"http\S+", "", text)

    @staticmethod
    def remove_stopwords(text, stopwords):
        words = [word.strip() for word in text.split() if word.strip().lower() not in stopwords]
        return " ".join(words)

    def denoise_text(self, text):
        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_urls(text)
        text = self.remove_stopwords(text, self.stopwords_all)
        return text

    def identify_language(self, text):
        try:
            detected_language = detect(text)
            return detected_language
        except LangDetectException:
            return np.nan

    @staticmethod
    def extract_emojis(text):
        return "".join(char for char in text if char in emoji.EMOJI_DATA)

    def process_messages(self, df, light_preprocessing=False):
        tqdm.pandas()

        only_url_regex = re.compile(r"http[s]*\S+$")
        has_url_regex = re.compile(r"http")

        df["text_length"] = df.text.progress_apply(len)
        df["text"] = df.text.progress_apply(lambda x: x.replace("\n", " "))

        df["only_url"] = df.text.progress_apply(lambda x: bool(only_url_regex.match(x)))
        df["has_url"] = df.text.progress_apply(lambda x: bool(has_url_regex.match(x)))

        df["text_cleaned"] = df["text"].str.lower()

        df["emoji"] = df["text"].progress_apply(self.extract_emojis)
        df["text_cleaned"] = df["text_cleaned"].progress_apply(emoji.demojize)

        df["language"] = df.progress_apply(
            lambda x: self.identify_language(x["text"]) if not x["only_url"] and x["text"] != "" else np.nan, axis=1
        )

        if light_preprocessing:
            df["text_cleaned"] = df["text_cleaned"].progress_apply(lambda x: self.denoise_text(x))
        else:
            df["text_cleaned"] = df["text_cleaned"].progress_apply(lambda x: self.process(x, light=False))

        df[["impressions", "reactions", "shares", "comments"]] = df[
            ["impressions", "reactions", "shares", "comments"]
        ].fillna(0)

        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['source_category'], random_state=random_state)
        return train_df, test_df