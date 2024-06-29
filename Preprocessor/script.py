import argparse
import pandas as pd
from .preprocessor import Preprocessor
from .paths import MESSAGES_PATH, TRAIN_PATH, TEST_PATH, FASTTEXT_PATH, STOPWORDS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process messages data.")
    parser.add_argument("--light", action="store_true", help="If set, uses light preprocessing.")
    
    args = parser.parse_args()
    
    print(f"Loading messages from: {MESSAGES_PATH}")
    print(f"Saving train data to: {TRAIN_PATH}")
    print(f"Saving test data to: {TEST_PATH}")

    # Load messages DataFrame
    messages_df = pd.read_json(MESSAGES_PATH, lines=True)
    
    # Initialize Preprocessor
    preprocessor = Preprocessor(model_path=FASTTEXT_PATH, stopwords_path=STOPWORDS_PATH)
    
    # Process messages
    processed_df = preprocessor.process_messages(messages_df, light_preprocessing=args.light)

    # Split the dataset into train and test sets
    train_df, test_df = preprocessor.split_data(processed_df)
    
    # Save the processed DataFrame
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    
    print("Data processing complete.")