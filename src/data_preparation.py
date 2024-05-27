import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import argparse

def load_dataset(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.columns = ["index_e", "English", "index_r", "Russian"]
    return data[["English", "Russian"]]

def char_tokenize(text):
    return list(text)

def build_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(char_tokenize(sentence))
    return {char: idx for idx, char in enumerate(sorted(vocab))}

def encode_sentence(sentence, vocab):
    return [vocab[char] for char in char_tokenize(sentence)]

def pad_sequences(sequences, maxlen, padding_value=0):
    return [seq + [padding_value] * (maxlen - len(seq)) for seq in sequences]

def preprocess_data(input_file, output_dir, max_length=100):
    df = load_dataset(input_file)
    vocab = build_vocab(df['English'].tolist() + df['Russian'].tolist())
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_en = [encode_sentence(sentence, vocab) for sentence in train_df['English'].tolist()]
    train_ru = [encode_sentence(sentence, vocab) for sentence in train_df['Russian'].tolist()]
    val_en = [encode_sentence(sentence, vocab) for sentence in val_df['English'].tolist()]
    val_ru = [encode_sentence(sentence, vocab) for sentence in val_df['Russian'].tolist()]

    train_en = pad_sequences(train_en, max_length)
    train_ru = pad_sequences(train_ru, max_length)
    val_en = pad_sequences(val_en, max_length)
    val_ru = pad_sequences(val_ru, max_length)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(train_en).to_csv(os.path.join(output_dir, 'train_en.csv'), index=False)
    pd.DataFrame(train_ru).to_csv(os.path.join(output_dir, 'train_ru.csv'), index=False)
    pd.DataFrame(val_en).to_csv(os.path.join(output_dir, 'val_en.csv'), index=False)
    pd.DataFrame(val_ru).to_csv(os.path.join(output_dir, 'val_ru.csv'), index=False)
    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preparation for Transformer Model')
    parser.add_argument(
        '--input_file', 
        type=str,
        default='../data/raw/rus_eng_sentences.tsv', 
        help='Path to the input CSV file')
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../data/processed/',
        help='Directory to save the processed data')
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=100, 
        help='Maximum length of sequences after padding')

    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_dir, args.max_length)