import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from Preprocessor import Preprocessor
from Preprocessor.paths import TEST_PATH, TRAIN_PATH, STOPWORDS_PATH, MODEL_PATH

test_df = pd.read_csv(TEST_PATH)
# Load the pre-trained RandomForest model
model_path = 'models/rf.pkl'
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)

preprocessor = Preprocessor(model_path=MODEL_PATH, stopwords_path=STOPWORDS_PATH)

tqdm.pandas()
test_df['hard_text'] = test_df['text'].progress_apply(lambda x: preprocessor.process(x, light=False))

with open('models/tfidf_vectorizer.pkl', 'rb') as file:
    vect = pickle.load(file)
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('models/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

num_features = ['impressions', 'reactions', 'shares', 'comments']
categorical_features = ['content_type', 'url_only']

test_text_tkz = vect.transform(test_df['hard_text'])
test_num = scaler.transform(test_df[num_features])
test_cat = encoder.transform(test_df[categorical_features])

X_test_processed = sp.hstack((test_text_tkz, test_num, test_cat), format='csr')

y_test_pred = rf_model.predict(X_test_processed)

with open('models/category_names.pkl', 'rb') as file:
    category_names = pickle.load(file)

y_test_pred_categories = [category_names[i] for i in y_test_pred]

def create_submission(df, numerical_predictions, filename='submission'):
    y_test_categories = [category_names[i] for i in numerical_predictions]
    df['source_category'] = y_test_categories
    category_counts = df.groupby('source_id')['source_category'].value_counts()
    majority_categories = category_counts.groupby(level=0).idxmax().apply(lambda x: x[1]).reset_index(name='source_category')
    print(f'Predicted source categories:\n{majority_categories.source_category.value_counts()}')
    majority_categories.to_csv(f'{filename}.csv', index=False)

create_submission(test_df, y_test_pred, filename='submission')

print("Inference completed and submission file created.")