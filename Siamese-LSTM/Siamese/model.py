from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Nadam
import wandb
import numpy as np
import time
import gc
import os
from inputHandler import create_train_dev_set



class SiameseBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length, number_lstm, number_dense, rate_drop_lstm, 
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio

    def train_model(self, sentences_pair, is_similar, embedding_meta_data, model_save_directory='./'):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass each sentence from sentences_pairs to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encoders and pass to a dense layer.
                3. Pass the dense layer vectors to sigmoid output layer.
                4. Use binary cross-entropy loss to train weights.
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory to save models

        Returns:
            best_model_path (str): path of the best model
        """
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']
        print("Embedding matrix shape:", embedding_matrix.shape)
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                            is_similar, self.max_sequence_length,
                                                                            self.validation_split_ratio)

        if train_data_x1 is None:
            print("Failure: Unable to train model")
            return None

        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(
                input_dim=len(tokenizer.word_index) + 1,  
                output_dim=self.embedding_dim,  
                weights=[embedding_matrix],  
                mask_zero=False, 
            )

        # Creating Bidirectional LSTM Encoder
        lstm_layer = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))

        # Input for the first sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        # Input for the second sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer(embedded_sequences_2)

        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))
        leaks_dense = Dense(int(self.number_dense_units/2), activation=self.activation_function)(leaks_input)

        # Merge LSTM encoded vectors and leaks dense layer
        merged = concatenate([x1, x2, leaks_dense])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1)(merged)

        # Define and compile the model
        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
        model.compile(loss='mean_squared_error', optimizer=Nadam(), metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        checkpoint_dir = os.path.join(model_save_directory, 'checkpoints', str(int(time.time())))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
        tensorboard = TensorBoard(log_dir=os.path.join(checkpoint_dir, "logs/{}".format(time.time())))

        # Train the model
        train_labels_scaled = np.array(train_labels)
        val_labels_scaled = np.array(val_labels)

        # Train the model
        history = model.fit([train_data_x1, train_data_x2, leaks_train], train_labels_scaled,
                    validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels_scaled),
                    epochs=20, batch_size=64, shuffle=True,
                    callbacks=[early_stopping, model_checkpoint, tensorboard, wandb.keras.WandbCallback()])
        print(history.history.keys(), " are the keys")
        # Log relevant information during training
        # wandb.log({'loss': history.history['loss'], 'val_loss': history.history['val_loss']})
        # wandb.log({'accuracy': history.history['accuracy'], 'val_accuracy': history.history['val_accuracy']})

    
        # print(history.history)
        # wandb.save('/content/drive/MyDrive/Siamese/best_model.h5')
        print("Model saved successfully!!")

        return bst_model_path


    def update_model(self, saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_sentences_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            model_path (str): model path of already trained siamese model
            new_sentences_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix

        Returns:
            return (best_model_path):  path of best model
        """
        tokenizer = embedding_meta_data['tokenizer']
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_mae', patience=3)

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels_scaled,
                validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels_scaled),
                epochs=200, batch_size=64, shuffle=True,
                callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path