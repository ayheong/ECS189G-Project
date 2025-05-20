import os
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return ' '.join(cleaned_tokens)


def clean_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)


def clean_all(base_dir='data/stage_4_data/text_classification'):
    subsets = ['train', 'test']
    labels = ['pos', 'neg']

    for subset in subsets:
        for label in labels:
            input_path = os.path.join(base_dir, subset, label)
            output_path = os.path.join(base_dir, subset, label)  # overwrite same folder
            print(f"Cleaning: {input_path} → {output_path}")
            clean_folder(input_path, output_path)

if __name__ == "__main__":
    clean_all()
