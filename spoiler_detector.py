import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Dense, Concatenate, Attention
from tensorflow.keras.models import Model
from check import generate_reference_report
from multiprocessing import Pool
import os

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', add_special_tokens=True, max_length=100)
def tokenize_batch(batch):
    return tokenizer(batch, truncation=True, padding='max_length', max_length=100, return_tensors='tf')

def train_bilstm():
    # Load data
    data = pd.read_json('./data/train_data.json', lines=True)
    print("Reading train data...")


    # Split the train set into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['review'].tolist(), data['is_spoiler'].values, test_size=0.1, random_state=42
    )

    # Tokenize the train and validation data
    num_processes = 4
    train_batches = np.array_split(train_texts, num_processes)
    val_batches = np.array_split(val_texts, num_processes)
    train_batches_str = [[str(text) for text in batch] for batch in train_batches]
    val_batches_str = [[str(text) for text in batch] for batch in val_batches]
    with Pool(num_processes) as p:
        train_encodings_list = list(tqdm(p.imap(tokenize_batch, train_batches_str), total=num_processes))
        val_encodings_list = list(tqdm(p.imap(tokenize_batch, val_batches_str), total=num_processes))

    train_encodings = {key: np.concatenate([enc[key] for enc in train_encodings_list], axis=0) for key in
                       train_encodings_list[0].keys()}
    val_encodings = {key: np.concatenate([enc[key] for enc in val_encodings_list], axis=0) for key in
                     val_encodings_list[0].keys()}

    train_input_ids = train_encodings['input_ids']
    val_input_ids = val_encodings['input_ids']

    # Define the Bi-LSTM model
    input_layer = Input(shape=(100,))
    embedding_layer = Embedding(input_dim=tokenizer.vocab_size, output_dim=100, input_length=100)(input_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    attention = Attention()([lstm_layer, lstm_layer])  # Self-attention
    attention_output = Concatenate()([lstm_layer, attention])
    dropout_layer = Dropout(0.2)(attention_output)
    lstm_output = Bidirectional(LSTM(32))(dropout_layer)
    dense_output = Dense(1, activation='sigmoid')(lstm_output)
    bilstm_model = Model(inputs=input_layer, outputs=dense_output)
    bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Train the model using the validation set for evaluation
    bilstm_model.fit(
        train_input_ids, train_labels,
        epochs=3, batch_size=32,
        validation_data=(val_input_ids, val_labels)
    )

    # Save the model
    bilstm_model.save("./models/bilstm_model")
    print("Saved model in ./models/bilstm_model")

    # Evaluate the model performance on the test dataset
    evaluate(bilstm_model)

def evaluate(model):
    # Load data
    data = pd.read_json('./data/test_data.json', lines=True)
    print("Reading test data...")

    num_processes = 4
    test_batches = np.array_split(data['review'].tolist(), num_processes)
    test_batches_str = [[str(text) for text in batch] for batch in test_batches]
    with Pool(num_processes) as p:
        train_encodings_list = list(tqdm(p.imap(tokenize_batch, test_batches_str), total=num_processes))

    test_encodings = {key: np.concatenate([enc[key] for enc in train_encodings_list], axis=0) for key in
                       train_encodings_list[0].keys()}

    y_pred = model.predict(test_encodings['input_ids'])

    y_pred = (y_pred > 0.5).astype(int)
    with open("./output/model_output.out", "w") as f:
        for pred in y_pred:
            f.write(f"{int(pred)}\n")

    # Save the actual test is_spoiler labels in test_reference
    with open("./data/reference/test_reference.txt", "w") as f:
        for label in data['is_spoiler'].values:
            f.write(f"{int(label)}\n")

    # Create report on the test dataset
    generate_reference_report()

def train_bilstm_with_plot():
    data = pd.read_json('./data/test_data.json', lines=True)
    print("Reading test data...")

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    train_review_texts = train_data['review'].tolist()
    train_plot_texts = train_data['summary'].tolist()
    val_review_texts = val_data['review'].tolist()
    val_plot_texts = val_data['summary'].tolist()

    train_review_encodings = tokenizer(train_review_texts, truncation=True, padding='max_length', max_length=100,
                                       return_tensors='tf')
    train_plot_encodings = tokenizer(train_plot_texts, truncation=True, padding='max_length', max_length=100,
                                     return_tensors='tf')

    val_review_encodings = tokenizer(val_review_texts, truncation=True, padding='max_length', max_length=100,
                                     return_tensors='tf')
    val_plot_encodings = tokenizer(val_plot_texts, truncation=True, padding='max_length', max_length=100,
                                   return_tensors='tf')

    train_review_input_ids = train_review_encodings['input_ids'].numpy()
    train_plot_input_ids = train_plot_encodings['input_ids'].numpy()
    val_review_input_ids = val_review_encodings['input_ids'].numpy()
    val_plot_input_ids = val_plot_encodings['input_ids'].numpy()

    # Define the Bi-LSTM model
    input_layer_review = Input(shape=(100,))
    input_layer_plot = Input(shape=(100,))

    embedding_layer_review = Embedding(input_dim=tokenizer.vocab_size, output_dim=100, input_length=100)(
        input_layer_review)
    lstm_layer_review = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer_review)
    dropout_layer_review = Dropout(0.2)(lstm_layer_review)
    lstm_output_review = Bidirectional(LSTM(32))(dropout_layer_review)

    embedding_layer_plot = Embedding(input_dim=tokenizer.vocab_size, output_dim=100, input_length=100)(input_layer_plot)
    lstm_layer_plot = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer_plot)
    dropout_layer_plot = Dropout(0.2)(lstm_layer_plot)
    lstm_output_plot = Bidirectional(LSTM(32))(dropout_layer_plot)

    # Concatenate the outputs from plot and review layers
    concatenated_output = Concatenate()([lstm_output_review, lstm_output_plot])
    dense_output = Dense(1, activation='sigmoid')(concatenated_output)
    bilstm_model = Model(inputs=[input_layer_review, input_layer_plot], outputs=dense_output)
    bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training on CPU.")
    bilstm_model.fit(
        [train_review_input_ids, train_plot_input_ids],  # Input for review and plot
        train_data['is_spoiler'].values,
        validation_data=([val_review_input_ids, val_plot_input_ids], val_data['is_spoiler'].values),
        epochs=5,
        batch_size=32
    )

def evaluate_with_plot(model):
    # Load data
    data = pd.read_json('./data/test_data.json', lines=True)
    print("Reading test data...")
    # -----------------------------------------
    # Consider the plot synopsis of the movie
    # -----------------------------------------

    test_review_texts = data['review'].tolist()
    test_review_encodings = tokenizer(test_review_texts, truncation=True, padding='max_length', max_length=100,
                                      return_tensors='tf')
    test_review_input_ids = test_review_encodings['input_ids'].numpy()

    test_plot_texts = data['summary'].tolist()
    test_plot_encodings = tokenizer(test_plot_texts, truncation=True, padding='max_length', max_length=100,
                                    return_tensors='tf')
    test_plot_input_ids = test_plot_encodings['input_ids'].numpy()

    y_pred = model.predict([test_review_input_ids, test_plot_input_ids])
    y_pred = (y_pred > 0.5).astype(int)
    with open("./output/model_output.out", "w") as f:
        for pred in y_pred:
            f.write(f"{int(pred)}\n")

    # Save the actual test is_spoiler labels in test_reference
    with open("./data/reference/test_reference.txt", "w") as f:
        for label in data['is_spoiler'].values:
            f.write(f"{int(label)}\n")

    # Create report on the test dataset
    generate_reference_report()

if __name__ == "__main__":
    if os.path.exists('models/bilstm_model'):
        print("Found model bilstm_model Evaluating...")
        evaluate(tf.keras.models.load_model('./models/bilstm_model'))
    else:
        print("Did not find model. Starting training...")
        train_bilstm()