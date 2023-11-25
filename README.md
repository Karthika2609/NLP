# NLP
DA

1) Naive Bayes Classifier using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis")
train_data = dataset["train"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    train_data["content"], train_data["polarity"], test_size=0.2, random_state=42
)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Classifier Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

2) Support Vector Machine (SVM) using sklearn
   from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Vectorize the text data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = make_pipeline(SVC(kernel="linear"))
svm_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Classifier Accuracy: {accuracy_svm}")
print(classification_report(y_test, y_pred_svm))

3) Bi-LSTM model using TensorFlow/Keras
    import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to have the same length
X_train_padded = pad_sequences(X_train_sequences, maxlen=50, padding="post")
X_test_padded = pad_sequences(X_test_sequences, maxlen=50, padding="post")

# Build a Bi-LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=50))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train_padded, y_train, epochs=5, validation_data=(X_test_padded, y_test))

# Evaluate the model
_, accuracy_lstm = model.evaluate(X_test_padded, y_test)
print(f"Bi-LSTM Model Accuracy: {accuracy_lstm}")

4)  transformer-based translation model
   # Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Masking, LSTM, Dense, Attention

# Download and preprocess the dataset
# You may need to customize the data loading and preprocessing based on the Samantar dataset structure.

# Sample code to load English to Hindi translation data
# Assume 'english_sentences' and 'hindi_sentences' are lists of English and Hindi sentences
# Make sure to adjust this based on your dataset format.

# Download the dataset from the provided link and extract it.
# Load the dataset into lists (english_sentences, hindi_sentences)

# Tokenize the sentences
english_tokenizer = keras.preprocessing.text.Tokenizer(filters="")
english_tokenizer.fit_on_texts(english_sentences)
english_seq = english_tokenizer.texts_to_sequences(english_sentences)

hindi_tokenizer = keras.preprocessing.text.Tokenizer(filters="")
hindi_tokenizer.fit_on_texts(hindi_sentences)
hindi_seq = hindi_tokenizer.texts_to_sequences(hindi_sentences)

# Pad sequences to a fixed length
max_len = max(len(english_seq[0]), len(hindi_seq[0]))
english_seq = keras.preprocessing.sequence.pad_sequences(english_seq, maxlen=max_len, padding="post")
hindi_seq = keras.preprocessing.sequence.pad_sequences(hindi_seq, maxlen=max_len, padding="post")

# Build the transformer model
def transformer_model(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, ff_dim, dropout=0.1):
    # Define the input layers
    encoder_inputs = Input(shape=(None,))
    decoder_inputs = Input(shape=(None,))

    # Embedding layers
    encoder_embedding = Embedding(vocab_size, d_model)
    decoder_embedding = Embedding(vocab_size, d_model)

    # Positional encoding layers
    encoder_positional_encoding = PositionalEncoding(max_len, d_model)
    decoder_positional_encoding = PositionalEncoding(max_len, d_model)

    # Apply the embedding and positional encoding layers
    x = encoder_positional_encoding(encoder_embedding(encoder_inputs))
    y = decoder_positional_encoding(decoder_embedding(decoder_inputs))

    # Transformer blocks
    for _ in range(num_encoder_layers):
        x = transformer_block(x, nhead, ff_dim, dropout)

    for _ in range(num_decoder_layers):
        y = transformer_block(y, nhead, ff_dim, dropout)

    # Output layer
    decoder_outputs = Dense(vocab_size, activation="softmax")(y)

    # Create the model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    return model

# Instantiate the model
model = transformer_model(
    vocab_size=VOCAB_SIZE,
    d_model=EMBEDDING_DIM,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT
)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy")

# Train the model
model.fit(
    x=[english_seq, hindi_seq[:, :-1]],
    y=hindi_seq.reshape(hindi_seq.shape[0], hindi_seq.shape[1], 1)[:, 1:],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

# Save the model
model.save("transformer_model.h5")

# Translate a sentence
def translate_sentence(sentence):
    # Preprocess the input sentence
    sentence = [english_tokenizer.word_index[word] for word in sentence.split(" ")]
    sentence = keras.preprocessing.sequence.pad_sequences([sentence], maxlen=max_len, padding="post")

    # Generate the translation
    translated_sentence = generate_translation(model, sentence, hindi_tokenizer)

    # Decode the translation
    translated_sentence = hindi_tokenizer.sequences_to_texts(translated_sentence)[0]

    return translated_sentence

# Sample translation
input_sentence = "Translate this sentence."
translation = translate_sentence(input_sentence)
print(f"Input: {input_sentence}")
print(f"Translation: {translation}")

   



   


