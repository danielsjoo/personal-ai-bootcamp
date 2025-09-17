# personal-ai-bootcamp
Just some AI problems to learn about neural networks.

The overall philosophy of this bootcamp is to build up pieces manually before I get to abstract them, meaning I will start with understanding then implementing mathematical foundations before adding them to my toolkit.

## Module 1: The Mathematical Foundation of Neural Networks
1. micrograd.ipynb
   1. Manual graph traversal and backpropogation
   2. Neuron unit
2. jax_xor.ipynb
   1. The first neural network. Solves the xor problem
   2. Matrix multiplications of MLP
   3. Manual data loader and training loop: full batching and SGD
   4. Mean squared error
3. jax_mnist.ipynb
   1. Abstracted data loader
   2. Abstracted optimizer: optax
   3. Softmax_cross_entropy loss

## Module 2: Motivation for Attention
4. jax_imdb_sentiment.ipynb
   1. word embeddings
   2. Sigmoid_binary_cross_entropy loss
   3. Global Average pooling
5. keras_mnist.ipynb
   1. Keras abstraction
6. keras imdb_mlp.ipynb
   1. Functions api and subclassing
   2. Pre-trained models
   3.  Validation loss, early stopping
   4.  Dropout layers
7.  keras_imdb_rnn.ipynb
    1.  Recurrent Neural Networks
    2.  Long Short Term Memory

## Module 3: Basics of Self Attention
7. keras_imdb_encoder.ipynb
   1. Multi-head attention
   2. Encoder block
8. keras_english_spanish.ipynb
   1. Decoder block, full transformer
   2.  Causal and padding masks

## Module 4: Convolutional Neural Networks
10. fashion_mnist_cnn.ipynb
    1.  Convolution layers
    2.  Batch normalization
11. cifar10_cnn.ipynb
    1.  Data augmentation
    2.  Resnet
    3.  Transfer learning

## Module 5: Scaling and Optimizing
12. pytorch_gpt2.ipynb
    1.  Decoder only transformers
    2.  Pytorch abstractions
    3.  Hugging Face datasets + tokenizers
    4.  Running remotely on HPC
13. cifar100_cnn.ipynb
    1.  tensorboard: visualizing metrics
    2.  distributing training

## Module 6: Advanced Attention - Alphafold
14.  evoformer.ipynb
     1.  Flax NNX / Jax again
     2.  Einstein Sumnation Notation
     3.  Gated Multi-head Attention
     4.  Axial Attention
     5.  Triangle Attention