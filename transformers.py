import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', FutureWarning)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K




# Data Preparation
# Sample parallel sentences (English -> Spanish)
input_texts = [
    "Hello.",
    "How are you?",
    "I am learning machine translation.",
    "What is your name?",
    "I love programming."
]

target_texts = [
    "Hola.",
    "¿Cómo estás?",
    "Estoy aprendiendo traducción automática.",
    "¿Cuál es tu nombre?",
    "Me encanta programar."
]

Output_texts = ["startseq " + x + " endseq" for x in target_texts]

# Now convert the text from the sentences to tokens and create a vocabulary
# Tokenization: Uses Tokenizer to convert words into numerical sequences.
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(Output_texts)
output_sequences = output_tokenizer.texts_to_sequences(Output_texts)

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

# Padding: Ensures all sequences have the same length.
# Padding
max_input_length = max([len(seq) for seq in input_sequences])
max_output_length = max([len(seq) for seq in output_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

# Prepare the target data for training
decoder_input_data = output_sequences[:, :-1]
decoder_output_data = output_sequences[:, 1:]

# Convert to one-hot
decoder_output_data = np.eye(output_vocab_size)[decoder_output_data]

print("input_sequences shape:    ", input_sequences.shape)
print("decoder_input_data shape: ", decoder_input_data.shape)
print("decoder_output_data shape:", decoder_output_data.shape)

# Self-Attention class
# In this implementation of self-attention layer:
#
#1.  We first initialize the weights in the build method, where:
# self.Wq, self.Wk, self.Wv are the trainable weight matrices.
# Their shape is (feature_dim, feature_dim), meaning they transform input features into Q, K, and V representations.
# 2. Applying Attention using call method. The call() method:
# Computes Q, K, V by multiplying inputs (encoder/decoder output) with their respective weight matrices.
# Computes dot-product attention scores using K.batch_dot(q, k, axes=[2, 2]), resulting in a (batch_size, seq_len, seq_len) matrix.
# Scales the scores to avoid large values.
# Applies softmax to normalize the attention scores.
# Multiplies attention weights with V to get the final output.
# 3. The compute_output_shape method defines the shape of the output tensor after the layer processes an input.
# The output shape of the Self-Attention layer remains the same as the input shape.
# The attention mechanism transforms the input but does not change its dimensions.4
# If the attention layer changed the shape, you would modify compute_output_shape

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
         # input_shape is a list: [q_shape, k_shape, v_shape]
         feature_dim = input_shape[0][-1]

         self.Wq = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wq'
         )

         self.Wk = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wk'
         )

         self.Wv = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wv'
         )
         super(SelfAttention, self).build(input_shape)

    def call(self, input):
        q, k, v = input

        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)

        # Scale dot-product attention
        scores = K.batch_dot(q, k, axes=[2, 2])
        dk = K.cast(K.shape(k)[-1], dtype=K.floatx())
        scores = scores / K.sqrt(dk)

        attention_weights = K.softmax(scores, axis=-1)
        output = K.batch_dot(attention_weights, v)

        return output

#   Model Architecture
# The model follows an Encoder-Decoder structure:
#
#   Encoder:
# Takes input sentences (padded and tokenized).
# Uses an Embedding layer (word representations) + LSTM (to process sequences).
# The LSTMs are used as the help process variable-length input sentences and generate meaningful translations.
# Outputs context vectors (hidden & cell states).
# Attention Layer
# Applied to both the encoder and decoder outputs.
# Helps the decoder focus on relevant words during translation.
#   Decoder
# Receives target sequences (shifted one step ahead).
# Uses an LSTM with encoder states as initial states.
# Applies self-attention for better learning.
# Uses a Dense layer (Softmax) to predict the next word.

# Encoder
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_sequences= True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(decoder_input_data.shape[1],))
decoder_embedding = Embedding(output_vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences= True, return_state=True)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention: decoder attends to encoder output
self_attention = SelfAttention()
attention_output = self_attention([
    decoder_outputs, encoder_outputs, encoder_outputs]
)

# combine decoder output with attention content
decoder_concat =Concatenate(axis=-1)([decoder_outputs, attention_output])

# final Dense Layer
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat)

# full Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Train the Model
history_glorot_adam = model.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)

# Plotting training loss
plt.plot(history_glorot_adam.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()