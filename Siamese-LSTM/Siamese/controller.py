import pandas as pd
from sklearn.model_selection import train_test_split
from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
from keras.models import load_model
from scipy.stats import pearsonr, spearmanr
import wandb

########################################
############ Data Preperation ##########
########################################
wandb.init(project='Semantic_textual_similarity', entity='bluehoax')

# Load the SICK dataset
df = pd.read_csv('/content/drive/MyDrive/Siamese/SICK.txt', delimiter='\t')

# Split data into train and dev sets
train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)

# Extract data from DataFrames (assuming column names)
train_sentences1 = train_df['sentence_A'].tolist()
train_sentences2 = train_df['sentence_B'].tolist()
train_relatedness_score = train_df['relatedness_score'].tolist()

# Development data
dev_sentences1 = dev_df['sentence_A'].tolist()
dev_sentences2 = dev_df['sentence_B'].tolist()
dev_relatedness_score = dev_df['relatedness_score'].tolist()

del df  # Clear memory

####################################
######## Word Embedding ############
####################################

tokenizer, embedding_matrix = word_embed_meta_data(train_sentences1 + train_sentences2, siamese_config['EMBEDDING_DIM'])
embedding_meta_data = {
    'tokenizer': tokenizer,
    'embedding_matrix': embedding_matrix
}

# Creating sentence pairs
train_sentences_pair = [(x1, x2) for x1, x2 in zip(train_sentences1, train_sentences2)]
dev_sentences_pair = [(x1, x2) for x1, x2 in zip(dev_sentences1, dev_sentences2)]

del train_sentences1, train_sentences2, dev_sentences1, dev_sentences2  # Clear memory

##########################
######## Training ########
##########################

# Model configuration
class Configuration(object):
    pass

CONFIG = Configuration()
CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

# Initialize and train the Siamese BiLSTM model
siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units, CONFIG.number_dense_units,
                        CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function,
                        CONFIG.validation_split_ratio)
best_model_path = siamese.train_model(train_sentences_pair, train_relatedness_score, embedding_meta_data,
                                      model_save_directory='./')


########################
###### Testing #########
########################

# Load the best model
model = load_model(best_model_path)

# Test the model
test_sentence1 = input("Enter the first sentence: ")
test_sentence2 = input("Enter the second sentence: ")
test_sentence_pair = [(test_sentence1, test_sentence2)]

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pair, siamese_config['MAX_SEQUENCE_LENGTH'])

preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
result = (test_sentence_pair[0], preds[0])

print(f"The similarity score is: {result[1]}")

# Evaluate on development set
dev_data_x1, dev_data_x2, leaks_dev = create_test_data(tokenizer, dev_sentences_pair, siamese_config['MAX_SEQUENCE_LENGTH'])
dev_preds = list(model.predict([dev_data_x1, dev_data_x2, leaks_dev], verbose=1).ravel())

# Calculate Spearman correlation
spearman_corr, _ = spearmanr(dev_preds, dev_relatedness_score)
print("Spearman correlation on dev:", spearman_corr)

# Calculate Pearson correlation
pearson_corr, _ = pearsonr(dev_preds, dev_relatedness_score)
print("Pearson correlation on dev:", pearson_corr)
bst_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

