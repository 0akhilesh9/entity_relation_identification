Implemented a system that identifies relations between entities as described in (https://pdfs.semanticscholar.org/6b8b/2075319accc23fef43e4cf76bc3682189d82.pdf)

core model is implemented in model.py

The embeddings matrix has dimensions vocab_size X embedding_size. The weight dimensions for attention calculation are hidden_size_RNN(128)*2 X 1.
Attention: The attention calculation is handled in a different function. The input to this function is the output of the RNN. The attention is calculated in three steps.
M_value: this is the hyperbolic tangent of the input values.
Alpha: This is the softmax of the product of the weights (omega) and the m_val
R_val: This is the reduced sum of the product of the alpha value and the input
H*value: Hyperbolic tangent of the r_val.


Forward pass: The inputs consists of tokens, and Pos tags of the tokens. First an embedding lookup is performed for the tokens and the POS tags. The input could be padded with some padding tokens and hence a mask is generated and fed to the Bi-GRU. The embeddings of the tokens and POS tags are concatenated and fed to the BiGRU layer. The output of the BiGRU is fed to the attention function. The result from the attention function is passed through the final Classification layer. The output from this layer is passed through a softmax for probability calculation for different relation labels.


Improvized version:

CNN based model is used for Relation Extraction.

The CNN architecture used consists of 4 CNN layers with 2,3,4,5 kernel sizes (left most dimension) and the input channel is one and the output channels are 128. For this task, the input matrix represents a batch of sentence: Each column of the matrix stores the feature information of the corresponding word. By applying a filter with a specific width the neighboring words are convolved. Afterwards, the results of the convolution layers are pooled through max pooling approach. In the end, the results are concatenated and fed into a final classification layer to predict a relation expressed in the sentence. The output from this classification layer is fed into the softmax to get the relative probabilities of the relations. The loss function used is cross entropy and the regularization is also implemented to prevent overfitting of the model.


## Train and Predict

The source code for this model is spread across util.py, train_lib.py, train_advanced.py, advanced_data.py, model.py, predict_advanced.py

The advanced_data.py is same as that of data.py but with an extra function for relative position info extraction and the read_instances, generate_batches funtions are modified to contain the relative position information.

The train_lib.py file is modified to include a flag paramter for running the advanced model or the base model.

The main model is implemented in the model.py file under 'MyAdvancedModel' class.

The predict_advanced.py file is same as predict.py except it contains load_trained_model function

The training of the model is done through train_advanced.py model by running the following command

#### Train a model
```
python train_advanced.py --embed-file embeddings/glove.6B.100D.txt --embed-dim 100 --batch-size 10 --num_epochs 10

# stores the model by default at : serialization_dirs/basic/
```
 
#### Predict with model
```
python predict_advanced.py --prediction-file advanced_test_prediction.txt --batch-size 10 --load-serialization-dir "serialization_dirs\advanced"
```
