#-------------
#souce code: https://github.com/Hvass-Labs/TensorFlow-Tutorials

#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
 
# load data
# ======================================================================

mark_start = 'ssss '
mark_end = ' eeee'

data_src = []
with open('data/dialogue1', 'r') as f:
    data_src = f.readlines()

data_dest = []
with open('data/dialogue2', 'r') as f:
    data_dest = [ mark_start+line.replace("\n", "")+mark_end for line in f]


# tokenizer
num_words = 10000

class TokenizerWrap(Tokenizer):
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        

        Tokenizer.__init__(self, num_words=num_words)

        
        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            
            truncating = 'pre'
        else:
            
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

     
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
       
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
       

        # Convert to tokens. 
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            truncating = 'pre'
        else:
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens


tokenizer_src = TokenizerWrap(texts=data_src,
                              padding='pre',
                              reverse=True,
                              num_words=num_words)

tokenizer_dest = TokenizerWrap(texts=data_dest,
                               padding='post',
                               reverse=False,
                               num_words=num_words)


tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded


token_start = tokenizer_dest.word_index[mark_start.strip()]
token_end = tokenizer_dest.word_index[mark_end.strip()]



# training data
# =============================================================

encoder_input_data = tokens_src


decoder_input_data = tokens_dest[:, :-1]
print(decoder_input_data.shape)

decoder_output_data = tokens_dest[:, 1:]
print(decoder_output_data.shape)




# create neural network
## create encoder
# ===============================================================
encoder_input = Input(shape=(None, ), name='encoder_input')
 # the length of the vectors output by the embedding-layer
embedding_size = 128
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')
# the size of the internal states of the Gated Recurrent Units (GRU)
state_size = 512
#3 GRU layers 
encoder_gru1 = GRU(state_size, name='encoder_gru1',return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',return_sequences=False)

def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input
    
    # Connect the embedding-layer.
    net = encoder_embedding(net)

    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)

    # This is the output of the encoder.
    encoder_output = net
    
    return encoder_output

encoder_output = connect_encoder()

## Create the Decoder
#thought vector
decoder_initial_state = Input(shape=(state_size,),name='decoder_initial_state')
# integer-tokens
decoder_input = Input(shape=(None, ), name='decoder_input')


decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1',return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',return_sequences=True)

decoder_dense = Dense(num_words,activation='linear',name='decoder_output')



def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output]) 

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)
model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])


##Loss Function
def sparse_cross_entropy(y_true, y_pred):

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean



optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])

##Train the Model


x_data = \
{
    'encoder_input': encoder_input_data,
    'decoder_input': decoder_input_data
}
y_data = \
{
    'decoder_output': decoder_output_data
}

validation_split = 10000 / len(encoder_input_data)
model_train.fit(x=x_data,
                y=y_data,
                batch_size=640,
                epochs=10,
                validation_split=validation_split)





# Translate Texts
# ==============================================================

def chatbot(input_text, true_output_text=None):

    # Convert the input-text to integer-tokens.
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    
    
    initial_state = model_encoder.predict(input_tokens)

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_dest.max_tokens

    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
   
    token_int = token_start

    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    while token_int != token_end and count_tokens < max_tokens:
        
        decoder_input_data[0, count_tokens] = token_int

        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        decoder_output = model_decoder.predict(x_data)
        
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        
        sampled_word = tokenizer_dest.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]
    
    # Print the user input
    print("Input text:")
    print(input_text)
    print()

    # Print the chatbot response
    print("Output text:")
    print(output_text)
    print()

    # Optionally print the true translated text
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()




chatbot(input_text="hi how are you",
          true_output_text='thanks fine')
