# DevkotaGPT

Implementation of Generatively Pretrained Transformer (GPT) based on Andrej Karpathy's [ Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) using Laxmi Prasad Devkota's poems.

# Implementations

1. **Development Code ( `gpt-dev.ipynb`)**
    - *Load Data and Build Vocabulary:* Read text from "lspd.txt," determine its length, and construct a character-level vocabulary.

    - *Implement Character Tokenization:* 
    Create mappings (stoi and itos) for character-to-index and index-to-character conversions using lambda functions.

    - *Utilize Subword Tokenization:* Employ the tiktoken library for subword tokenization which is similar to OpenAI's GPT, showcasing encoding and decoding operations.

    - *Encode Data Efficiently:* Convert the entire text dataset into Torch tensors using the character-level tokenizer, for training and validation splits.

    - *Streamline Batch Processing:* Develop a function to generate batches of data with random indexing for specified block sizes (block_size) for training data.

    - *Build Bigram Language Model:* Define a basic language model (`BigramLanguageModel`) utilizing an embedding table for token embeddings, predicting subsequent tokens based on context.

    - *Train with Optimization:* Implement a training loop using AdamW optimizer and Cross Entropy Loss for optimizing model performance.

    - *Explore Self-Attention Mechanisms:* Delve into self-attention concepts through matrix operations, calculating attention weights and applying them using linear transformations and masking techniques.

1. **Tokenization & Word Embedding (`tokenizers_embedding.ipynb`)**
    The notebook provides a hands-on introduction to tokenization of Nepali text, loading and utilizing pre-trained word embeddings, and performing vector operations on Nepali words. 
    It combines theoretical explanations with practical examples using `nepalitokenizers` and `gensim` for natural language processing tasks in Nepali.
    
    You can download the model `nepali_embeddings_word2vec.txt` from [IEEE Dataport](https://ieee-dataport.org/open-access/300-dimensional-word-embeddings-nepali-language)


1. **Basic Bigram Language Model (`bigram.py`)**: 
    Basic bigram model with token and position embeddings, followed by a simple feedforward network.
    
    Components
    - Token Embeddings
    - Position Embeddings
    - Simple feedforward neural network for prediction

1. **Introduction to Multi-Head Self-Attention (`attention-applied.py`)**:
    Introduced multi-head self-attention for better sequence modeling.

    Components
    - Token Embeddings
    - Position Embeddings
    - Multi-Head Self-Attention (4 heads)
    - Linear layer for prediction

1. **Addition of FeedForward Layer (`feed-forward.py`)**:
    Added a feedforward layer post-attention to capture non-linear dependencies.

    Components
    - Token Embeddings
    - Position Embeddings
    - Multi-Head Self-Attention (4 heads)
    - FeedForward layer (Linear layer + ReLU)
    - Linear layer for prediction

1. **Transformer Blocks with Residual Connections (`block.py`)**:
    Implemented stacked Transformer Blocks with residual connections, combining multi-head attention and feedforward layers for a deeper and more robust architecture.

    Components
    - Token Embeddings
    - Position Embeddings
    - Stacked Transformer Blocks (3 blocks):
        - Each block contains:
            - Multi-Head Self-Attention (4 heads)
            - FeedForward layer (Linear layer + ReLU + Linear layer)
            - Residual connections around both the attention and feedforward components
    - Linear layer for prediction


# Usage
1. For `.ipynb` notebooks you can run each cell and check the output
2. For `.py` files, you can simply run with python
    ```bash
    python bigram.py
    ```


