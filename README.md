# Semantic-texual-similarity
Aiming to evaluate and compare the performance of four different models in determining the similarity between sentence pairs

This repository contains implementations and evaluations of three different methods for testing Semantic Textual Similarity (STS). The goal of STS tasks is to measure the semantic similarity between pairs of sentences, where each method provides its unique approach to the problem.

## Repository Structure

```
├── SentEval.ipynb               # Notebook for testing STS with SentEval
├── BERT_STS.ipynb               # Notebook for testing STS with BERT
├── Siamese-LSTM.zip             # Zipped folder containing the Siamese-LSTM model files
└── README.md                    # Project documentation
```

## Methods for Testing STS

We have implemented three approaches for STS:

1. **SentEval Framework**  
   - The `SentEval.ipynb` notebook uses the [SentEval framework](https://github.com/facebookresearch/SentEval) to evaluate sentence embeddings on various similarity tasks.
   - SentEval provides a flexible and reliable evaluation framework, allowing you to assess different sentence encoders' performance on STS tasks.

2. **BERT-Based STS**  
   - The `BERT_STS.ipynb` notebook utilizes BERT for STS tasks.
   - This approach uses BERT's pre-trained transformer model to encode sentences into vector representations, which are then compared to determine similarity.
   - BERT's language model-based embeddings are robust for various natural language understanding tasks, including semantic similarity.

3. **Siamese LSTM for STS**  
   - The `Siamese-LSTM.zip` file contains the code and resources required to implement a Siamese LSTM model.
   - This model employs two LSTMs (Long Short-Term Memory networks) that work in parallel to encode two input sentences, capturing their semantic meanings and then comparing them to calculate similarity.

## Requirements

To run these models, make sure you have the following libraries installed:

- `pytorch`
- `transformers`
- `tensorflow`
- `numpy`
- `pandas`
- `sklearn`

You can install these packages using:
```bash
pip install torch transformers tensorflow numpy pandas scikit-learn
```

## Usage

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **SentEval**: Open and run `SentEval.ipynb` to test STS with the SentEval framework.

3. **BERT STS**: Run `BERT_STS.ipynb` to test BERT embeddings on STS tasks.

4. **Siamese LSTM**:
   - Unzip `Siamese-LSTM.zip`.
   - Follow the README within the unzipped folder for running the Siamese LSTM model.

## Results

The results from each STS approach can be found in the respective notebook outputs. Each method outputs similarity scores, which you can use to compare the performance of different models on the STS task.
