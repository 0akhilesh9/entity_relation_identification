import tensorflow as tf
from tensorflow.keras import layers, models, Sequential

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):
    # Initialization
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()
        # Initialize variables
        self.num_classes = len(ID_TO_CLASS)
        # Final classification layer
        self.decoder = layers.Dense(units=self.num_classes)
        # Weights for alpha calculation during attention
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        # Word embeddings
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        # Forward GRU layer
        self.forward_gru_op = layers.GRU(hidden_size, activation='tanh', recurrent_activation='sigmoid', implementation=2, return_sequences=True, return_state=False, go_backwards=False)
        # Backward GRU layer
        self.backward_gru_op = layers.GRU(hidden_size, activation='tanh', recurrent_activation='sigmoid', implementation=2, return_sequences=True, return_state=False, go_backwards=True)
        # Birectional layer with GRU layers
        self.bidir_gru = layers.Bidirectional(layer=self.forward_gru_op, merge_mode="concat", backward_layer=self.backward_gru_op)

    # Attention calculation
    def attn(self, rnn_outputs):
    # Followed paper(Attention-based bidirectional long short-term memory networks for relation classification.) terminology
        
        # Take tan hyperbolic of the rnn_outputs
        m_val = tf.math.tanh(rnn_outputs)
        # Multiply this value with the omega weights defined before
        alpha = tf.nn.softmax(tf.tensordot(m_val, self.omegas, axes=1), 1)
        # alpha = tf.expand_dims(tf.nn.softmax(tmp_val), -1)

        # Multiply the rnn_outputs with the alpha value
        r_val = tf.math.reduce_sum(tf.multiply(rnn_outputs, alpha), axis = 1)
        # Take the tan hyperbolic of the result
        h_star_val = tf.math.tanh(r_val)
        
        return h_star_val

    # Forward pass
    def call(self, inputs, pos_inputs, training):
        # Obtain the wordembeddings
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        # Fetch the POS embeddings
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        # Concatentate the word and POS embeddings
        embed_val = tf.concat([word_embed, pos_embed], axis=2)
        
        # Masking to remove padded tokens
        tokens_mask = tf.cast(inputs!=0, tf.float32)
        
        # import pdb
        # pdb.set_trace()
        
        # L-R GRU pass 
        #forward_gru_op = self.forward_gru_op(embed_val)
        # R-L GRU pass
        #backward_gru_op = self.backward_gru_op(embed_val)
        # Concatentate the results of the two GRU layers
        # bigru_op = tf.concat([forward_gru_op, backward_gru_op], axis=2)
        # Bidirectional GRU pass
        bigru_op = self.bidir_gru(embed_val, mask=tokens_mask)
        # Perform attention mechanism
        attn_op = self.attn(bigru_op)
        # Pass through the final Classification layer
        logits = self.decoder(attn_op)
        
        return {'logits': logits}

class MyAdvancedModel(models.Model):
    # Initialization
    def __init__(self, params):
        super(MyAdvancedModel, self).__init__()
        # define embeddings lookup
        self.embeddings = tf.Variable(tf.random.normal((params["vocab_size"], params["embed_dim"])))  #nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)
        # positional embeddings for entity 1
        self.pos1_embedding = tf.Variable(tf.random.normal((params["pos_dis_limit"] * 2 + 3, params["pos_emb_dim"])))  #nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        # positional embeddings for entity 2
        self.pos2_embedding = tf.Variable(tf.random.normal((params["pos_dis_limit"] * 2 + 3, params["pos_emb_dim"])))  #nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

        
        # dropout layer
        self.dropout = layers.Dropout(params["dropout_ratio"])
        
        # Feature dimensions for the CNN layer
        feature_dim = params["word_emb_dim"] * 2  + params["pos_emb_dim"] * 2
        
        # self.omegas = tf.Variable(tf.random.normal((len(params["filters"])*params["filter_num"], params["filter_num"])))
        
        # Define CNN layers
        self.conv_layers = []
        for k_size in params["filters"]:
          self.conv_layers.append(layers.Conv2D(params["filter_num"], (k_size, feature_dim)))
        
        # number of prediction classes
        self.num_classes = len(ID_TO_CLASS)

        # output Classification layer
        self.linear = layers.Dense(units=self.num_classes)
        
    # Attention calculation
    def attn(self, cnn_outputs):
    # Followed paper(Attention-based bidirectional long short-term memory networks for relation classification.) terminology
        
        # Take tan hyperbolic of the cnn_outputs
        m_val = tf.math.tanh(cnn_outputs)
        # Multiply this value with the omega weights defined before
        alpha = tf.nn.softmax(tf.tensordot(m_val, self.omegas, axes=1), 1)
        # alpha = tf.expand_dims(tf.nn.softmax(tmp_val), -1)

        # Multiply the cnn_outputs with the alpha value
        r_val = tf.math.reduce_sum(tf.multiply(cnn_outputs, alpha), axis = 1)
        # Take the tan hyperbolic of the result
        h_star_val = tf.math.tanh(r_val)
        
        return h_star_val
     
    # Forward pass
    def call(self, inputs, pos_inputs, pos1, pos2, training):
        # Lookup word embeddings
        word_embs = tf.nn.embedding_lookup(self.embeddings, inputs)
        # Lookup POS embeddings
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        # Lookup positional embeddings for entity 1
        pos1_embs = tf.nn.embedding_lookup(self.pos1_embedding, pos1)
        # Lookup positional embeddings for entity 2
        pos2_embs = tf.nn.embedding_lookup(self.pos2_embedding, pos2)
        
        # Concatentate all the embeddings to get input feature
        input_feature = tf.concat([word_embs, pos_embed, pos1_embs, pos2_embs], axis=2)     # batch_size X tokens X final_embedding_size(220)

        # import pdb  
        # pdb.set_trace()
        
        # Expand the dimensions to 4d to batch_size, Height, Width, Channels to feed to CNN layers
        in_x = tf.expand_dims(input_feature, 3)    # NHWC   batch_size X tokens X final_embedding_size(220) X 1(input channels = 1)
        # Dropout layer
        in_x = self.dropout(in_x)
        # Pass through CNN layers
        in_x = [tf.squeeze(tf.math.tanh(conv(in_x)), 2) for conv in self.conv_layers]
        # Max pool layer
        in_x = [tf.squeeze(tf.nn.max_pool1d(i, i.shape[1], strides=1, padding="VALID"), 1) for i in in_x]
        # Concatentate all the outputs
        in_x_features = tf.concat(in_x, axis=1)
        # Dropout layer for regularization
        x = self.dropout(in_x_features)
        # Final pass through the Classification layer
        logits = self.linear(x)

        return {'logits': logits}