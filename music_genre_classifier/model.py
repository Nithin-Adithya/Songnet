import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, MaxPool1D, GlobalAveragePooling1D, Dense, TimeDistributed, LSTM, GRU, Layer, Multiply, Permute, Reshape, Lambda, RepeatVector
from tensorflow.keras import backend as K

class SelfAttention(Layer):
    """
    Bahdanau-style attention layer.
    """
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        
        # e = W*x + b
        # We compute scores for each time step
        e = K.tanh(K.dot(x, self.W) + self.b)
        # e shape: (batch_size, time_steps, 1)
        
        # Remove singleton dim for softmax
        e = K.squeeze(e, axis=-1) 
        # e shape: (batch_size, time_steps)
        
        alpha = K.softmax(e)
        # alpha shape: (batch_size, time_steps)
        
        # Weighted sum
        # Reshape alpha for broadcasting
        alpha = K.expand_dims(alpha, axis=-1)
        
        output = x * alpha
        # output shape: (batch_size, time_steps, features) but weighted
        
        # Sum over time to get context vector (if we were doing pure attention pooling)
        # But SongNet paper says "TimeDistributed Dense" -> "Mean over time". 
        # If we use Attention, we usually replace "Mean over time" with "Weighted Sum over time".
        
        return K.sum(output, axis=1) # (batch_size, features)

def create_songnet_model(input_shape, num_classes, rnn_type='gru', use_attention=True):
    inputs = Input(shape=input_shape)
    
    # Conv Block 1
    x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D(pool_size=2)(x)
    
    # Conv Block 2
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D(pool_size=2)(x)
    
    # Conv Block 3
    x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D(pool_size=2)(x)
    
    # Recurrent Layer
    if rnn_type.lower() == 'lstm':
        x = LSTM(128, return_sequences=True)(x)
    elif rnn_type.lower() == 'gru':
        x = GRU(128, return_sequences=True)(x)
    else:
        raise ValueError("d_type must be 'lstm' or 'gru'")
    
    if use_attention:
        # Attention Pooling (replaces TimeDistributed + Mean)
        # Or place it after TimeDistributed?
        # Usually Attention is applied on RNN outputs directly.
        
        # Let's apply Attention on RNN sequence
        x = SelfAttention()(x) # Returns (batch, 128)
        
        # Maybe add a dense after attention?
        x = Dense(128, activation='relu')(x)
        
    else:
        # Original SongNet approach
        # TimeDistributed Dense
        x = TimeDistributed(Dense(128, activation='relu'))(x)
        # Mean over time (Global Average Pooling)
        x = GlobalAveragePooling1D()(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model_name = f"SongNet_{rnn_type.upper()}{'_Attn' if use_attention else ''}"
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    return model

if __name__ == "__main__":
    model = create_songnet_model((1292, 128), 8, rnn_type='gru', use_attention=True)
    model.summary()
