import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, Embedding, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import unicodedata

tf.random.set_seed(42)
np.random.seed(42)

EMBEDDING_DIM = 256
HIDDEN_STATE_DIM = 512
CELL_TYPE = 'LSTM'
BATCH_SIZE = 64
EPOCHS = 30
VALIDATION_SPLIT = 0.2

filepaths = [
    '/content/bn.translit.sampled.dev.tsv',
    '/content/bn.translit.sampled.test.tsv', 
    '/content/bn.translit.sampled.train.tsv'
]

def load_and_preprocess_data(filepaths, sample_size=5000):
    dfs = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, sep='\t', header=None, names=['latin', 'devanagari'])
        df['latin'] = df['latin'].astype(str)
        df['devanagari'] = df['devanagari'].astype(str)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    if sample_size is not None:
        combined_df = combined_df.sample(min(sample_size, len(combined_df)), random_state=42)
    
    def normalize(text):
        text = str(text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text.lower()
    
    combined_df['latin'] = combined_df['latin'].apply(normalize)
    combined_df['devanagari'] = combined_df['devanagari'].apply(lambda x: '\t' + str(x) + '\n')
    
    return combined_df['latin'].values, combined_df['devanagari'].values

latin_text, devanagari_text = load_and_preprocess_data(filepaths)

def tokenize(texts):
    tokenizer = Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer, max_len

input_sequences, input_tokenizer, max_input_len = tokenize(latin_text)
target_sequences, target_tokenizer, max_target_len = tokenize(devanagari_text)

input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

x_train, x_test, y_train, y_test = train_test_split(
    input_sequences, target_sequences, test_size=0.2, random_state=42
)

def build_seq2seq_model(input_vocab_size, target_vocab_size, max_input_len, max_target_len,
                       embedding_dim=EMBEDDING_DIM, hidden_state_dim=HIDDEN_STATE_DIM, cell_type=CELL_TYPE):
    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    
    if cell_type == 'LSTM':
        encoder_rnn = LSTM(hidden_state_dim, return_state=True)
    elif cell_type == 'GRU':
        encoder_rnn = GRU(hidden_state_dim, return_state=True)
    else:
        encoder_rnn = SimpleRNN(hidden_state_dim, return_state=True)
    
    encoder_outputs, *encoder_states = encoder_rnn(encoder_embedding)
    
    decoder_inputs = Input(shape=(max_target_len-1,))  # Changed to max_target_len-1
    decoder_embedding = Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    
    if cell_type == 'LSTM':
        decoder_rnn = LSTM(hidden_state_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)
    elif cell_type == 'GRU':
        decoder_rnn = GRU(hidden_state_dim, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states[0])
    else:
        decoder_rnn = SimpleRNN(hidden_state_dim, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states[0])
    
    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = TimeDistributed(decoder_dense)(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

model = build_seq2seq_model(
    input_vocab_size,
    target_vocab_size,
    max_input_len,
    max_target_len,
    embedding_dim=EMBEDDING_DIM,
    hidden_state_dim=HIDDEN_STATE_DIM,
    cell_type=CELL_TYPE
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

decoder_input_data = y_train[:, :-1]  # All except last token
decoder_target_data = y_train[:, 1:]  # All except first token

val_decoder_input_data = y_test[:, :-1]
val_decoder_target_data = y_test[:, 1:]

history = model.fit(
    [x_train, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=([x_test, val_decoder_input_data], np.expand_dims(val_decoder_target_data, -1)),
    verbose=1
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()