# End to End RNN(LSTM) project: IMDB movie review Sentiment analysis project

📖 Project Overview

This project implements a sentiment analysis model using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers to classify IMDB movie reviews as positive or negative.
The objective is to build an end-to-end deep learning pipeline covering preprocessing, model training, evaluation, and prediction.

🎯 Motivation

Understanding user sentiment from text is a core problem in NLP, widely used in:

Customer support automation

Recommendation systems

Opinion mining

Market research

Traditional bag-of-words or TF-IDF approaches ignore word order.
RNNs (especially LSTMs) capture sequential dependencies in text, making them ideal for sentiment classification.


📂 Dataset

IMDb Movie Reviews Dataset (available via TensorFlow / Keras).

50,000 labeled movie reviews

25,000 training, 25,000 testing

Balanced dataset: 💬 50% positive, 50% negative

Pre-tokenized with integer encoding

🧠 Model Architecture

The model uses an Embedding → LSTM → Dense pipeline.

Key Components

Embedding Layer

Converts input tokens into continuous dense vectors.

LSTM Layer

Captures long-term dependencies

Handles sequential context of sentences

Fully Connected Layer

Maps LSTM output to binary sentiment prediction

Sigmoid Output

Probability between 0 and 1 (positive vs negative)

🔬 Evaluation Insights

Reviews with stronger linguistic polarity are easily classified.

LSTM captures context like “not good at all”:  negative

Handling sarcasm is challenging:

“The movie was so boring that it was amazing I stayed awake.”
