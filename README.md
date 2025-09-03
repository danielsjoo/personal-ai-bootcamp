# personal-ai-bootcamp
Just some AI problems to learn about neural networks

In this AI bootcamp, the exercises go like this with each one building on the previous:
1. micrograd.ipynb
   1. Build the forward pass with numpy and manually compute gradients
   2. Build a nueron which outputs a weighted sum with these tracked values
2. jax_xor.ipynb
   1. The first neural network. Solves the xor problem
   2. Express the nueral network as a series of explicitly defined matrix multiplications and activation functions
   3. Manually implement the data loader and training loop: full batching and SGD
   4. mean squared error
3. jax_mnist.ipynb
   1. Use more libraries like optax and a data loader
   2.softmax_cross_entropy loss
4. jax_imdb_sentiment.ipynb
   1. introduction to word embeddings
   2. sigmoid_binary_cross_entropy loss
   3. Global Average pooling
5. keras imdb_mlp.ipynb
   1. The keras level abstractions with functions api and subclassing
   2. pre-trained models
   3. validation loss, using it for early stopping
   4. dropout layers
6. keras_imdb_rnn.ipynb
   1. Recurrent Neural Networks
   2. Long Short Term Memory
   3. Keras subclassing and implementing the above
7. keras_imdb_transformer.ipynb
   1. Basic transformer encoder block
   2. implementation of multi-head attention with mask

Later steps (hopefully):
1. English to Spanish translation with the encoder + decoder transformer
2. CNNs and Resnets image processing (probably a few modules)
3. Axial Attention (probably a few modules)
4. Graph Neural Networks (probably a few modules)