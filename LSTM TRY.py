## ------------------------------------------------------------------------------
#CREATE A PANDAS DF WITH THE ORIGINAL DATA
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.tokens import Doc, Span
from spacy.matcher import DependencyMatcher
from spacy.matcher import Matcher
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input, concatenate, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import urllib.request
import zipfile
import os
import mpld3


#create a pandas df with the original .ann file
def create_dataframe(content):
    with open(content, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.split() for line in lines]
    df = pd.DataFrame(data, columns=['token', 'tag'])
    df = df.drop(index = df.index[0])
    return df

df_file1 = create_dataframe('file1.ann')


def pulisci_df(df):
    def is_arabic(word):
        if word is None or pd.isnull(word):  # Check for None or NaN
            return True
        word = str(word)
        #remove_chars = "\n" + string.punctuation  + "\”" + "\“" + "\’" + "“”‘’–—…«»‹›"
        #if word in remove_chars:
            #return True
        return any('\u0600' <= char <= '\u06FF' for char in word)

    mask = df['token'].apply(is_arabic)
    return df[~mask]  # Keep rows where the condition is False

df_file1 = pulisci_df(df_file1)
df_file1 = df_file1.reset_index(drop=True)

## ------------------------------------------------------------------------------
#CREATE A SPACY DF

def pulisci_file(df, column_name):
    result = df[column_name].str.cat(sep=' ')
    return result

file1_text = pulisci_file(df_file1, 'token')

doc1 = nlp(file1_text)

##
row_to_insert1 = [".", "O"]
row_to_insert2 = [".", "I-Location"]
row_to_insert3 = [".", "I-Quantity"]
row_to_insert4 = [".", "I-Temporal"]
row_to_insert5 = [".", "I-Person"]

# The indexes where rows need to be inserted
indexes_to_insert = {239, 946, 981, 2165, 2298 #I-Location
                     , 3108, #I-Location,
                     3487, 3674, #I_quantity,
                     3929, 4207, 7248,
                     7988, 8201, #I_quantity,
                     11567, 11741, 13590, #I_quantity
                     13593, #I_quantity
                     13943,#O
                     14364, #I_quantity
                     14830,#I_Location
                     16391, #I_Temporal
                     16592, #I_quantity
                     17136, 18007, 19293}

# Insert the rows and reset index
for index in sorted(indexes_to_insert):
    if index == 13943:
        new_row = pd.DataFrame([row_to_insert1], columns=df_file1.columns)
    elif index in {2298, 3108, 14830}:
        new_row = pd.DataFrame([row_to_insert2], columns=df_file1.columns)
    elif index in {3674, 13590, 13593, 14364, 16592}:
        new_row = pd.DataFrame([row_to_insert3], columns=df_file1.columns)
    elif index == 16391:
        new_row = pd.DataFrame([row_to_insert4], columns=df_file1.columns)
    else:
        new_row = pd.DataFrame([row_to_insert5], columns=df_file1.columns)

    df_file1 = pd.concat([df_file1.iloc[:index], new_row, df_file1.iloc[index:]], ignore_index=True)


def extract_entities(df, tag_column="tag"):
    entities = []
    current_entity = None

    for index, row in df.iterrows():
        tag = row[tag_column]

        if tag.startswith("B-"):  # Begin a new entity
            if current_entity:  # Save the previous entity
                entities.append(current_entity)
            entity_type = tag[2:]  # Extract the entity type
            current_entity = {"type": entity_type, "start": index, "end": index + 1}

        elif tag.startswith("I-") and current_entity:  # Inside an entity
            entity_type = tag[2:]
            if current_entity["type"] == entity_type:  # Ensure it's the same entity type
                current_entity["end"] = index + 1  # Extend the current entity

        else:  # Outside any entity
            if current_entity:  # Save the current entity
                entities.append(current_entity)
                current_entity = None

    # Add the last entity if it exists
    if current_entity:
        entities.append(current_entity)

    return entities

# Extract entities
custom_entities = extract_entities(df_file1)

doc1.ents = [Span(doc1, entity["start"], entity["end"], label=entity["type"]) for entity in custom_entities]
doc1.set_ents(doc1.ents, default="outside")

##
data = []
for token in doc1:
    data.append([
        getattr(token, "text"),      # Token text
        getattr(token, "lemma_", None),   # Lemma
        getattr(token, "pos_", None),     # Part-of-speech tag
        getattr(token, "dep_", None),     # Dependency label
        getattr(token.head, "text", None),# Head text
        f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"
    ])


def join_and_clear(df, col_name, row1, row2):
    df.at[row1, col_name] = f"{df.at[row1, col_name]} {df.at[row2, col_name]}"
    df.loc[row2] = [None] * len(df.columns)

df = pd.DataFrame(data, columns=["token", "Lemma", "POS", "DEP", "Head", "tag"])


## ------------------------------------------------------------------------------
#PATTERNS

matcher = Matcher(nlp.vocab)
dependency_matcher = DependencyMatcher(nlp.vocab)

pattern_temporal = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "organization", "RIGHT_ATTRS": {"ENT_TYPE": "Organisation"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "date", "RIGHT_ATTRS": {"ENT_TYPE": "Temporal"}}
]

pattern_affiliation_to_org = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "person", "RIGHT_ATTRS": {"ENT_TYPE": {"in": ["Person", "Nationality"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "org", "RIGHT_ATTRS": {"ENT_TYPE": "Organisation"}}
]

pattern_location_operation = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "person", "RIGHT_ATTRS": {"ENT_TYPE": {"in": ["Person", "Organisation"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "location", "RIGHT_ATTRS": {"ENT_TYPE": "Location"}}
]

pattern_location_operation2 = [
    {"RIGHT_ID": "location", "RIGHT_ATTRS": {"ENT_TYPE": "Location"}},
    {"LEFT_ID": "location", "REL_OP": ".*", "RIGHT_ID": "person", "RIGHT_ATTRS": {"ENT_TYPE": {"in": ["Person", "Organisation"]}}}
]

pattern_cooperation = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "lemma", "RIGHT_ATTRS": {"LEMMA": {"in": ["with", "support", "cooperation", "condolence"]}}},
    {"LEFT_ID": "lemma", "REL_OP": ">>", "RIGHT_ID": "org1", "RIGHT_ATTRS": {"ENT_TYPE": {"in":["Organisation", "Location"]} }},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "org2", "RIGHT_ATTRS": {"ENT_TYPE": {"in":["Organisation", "Location"]}}}
]

pattern_cooperation2 = [
    {"RIGHT_ID": "lemma", "RIGHT_ATTRS": {"LEMMA": {"in": ["with", "support", "cooperate", "condolence", "between", "relation"]}}},
    {"LEFT_ID": "lemma", "REL_OP": ">>", "RIGHT_ID": "org1", "RIGHT_ATTRS": {"ENT_TYPE": {"in":["Organisation", "Location"]}}},
    {"LEFT_ID": "org1", "REL_OP": ">>", "RIGHT_ID": "org2", "RIGHT_ATTRS": {"ENT_TYPE": {"in":["Organisation", "Location"]}}}
]

pattern_cooperation3 = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"in": ["support", "cooperate", "help"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "org1", "RIGHT_ATTRS": {"DEP": "nsubj", "ENT_TYPE": {"in":["Organisation", "Location"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "org2", "RIGHT_ATTRS": {"ENT_TYPE": {"in":["Organisation", "Location"]}}}
]

pattern_compensation = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"ENT_TYPE": {"in": ["Person", "Organisation"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "money", "RIGHT_ATTRS": {"ENT_TYPE": "Money"}}
]

pattern_weapons_operation = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "weapon", "RIGHT_ATTRS": {"ENT_TYPE": {"in": ["MilitaryPlatform", "Weapon"]}}}
]


pattern_attack_against = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"ENT_TYPE": "Organisation"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "against", "RIGHT_ATTRS": {"LEMMA": "against"}},
    {"LEFT_ID": "against", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"ENT_TYPE": "Organisation"}}
]

pattern_attack_against2 = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"in": ["defeat", "kill", "destroy"]}}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj", "ENT_TYPE": "Organisation"}},
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "org", "RIGHT_ATTRS": {"ENT_TYPE":{"in": ["Organisation", "Location"]}}},
]

pattern_founded_by = [
    {"POS": {"in": ["NOUN", "PROPN"]}, "OP": "?"},
    {"LEMMA": "found", "POS": "VERB"},
    {"LOWER": {"IN": ["by", "in"]}},
    {"POS": {"IN": ["PROPN", "NOUN"]}, "ENT_TYPE": {"IN": ["Person", "Organisation", "Temporal"]}, "OP": "?"}
]

dependency_matcher.add("TEMPORAL", [pattern_temporal])
dependency_matcher.add("AFFILIATION_TO_ORG", [pattern_affiliation_to_org])
dependency_matcher.add("LOCATION_OF_OPERATION", [pattern_location_operation, pattern_location_operation2])
dependency_matcher.add("COOPERATION", [pattern_cooperation, pattern_cooperation2])
dependency_matcher.add("COMPENSATION", [pattern_compensation])
dependency_matcher.add("WEAPONS_OF_OPERATION", [pattern_weapons_operation])
dependency_matcher.add("ATTACK", [pattern_attack_against, pattern_attack_against2])
matcher.add("FOUNDED_BY", [pattern_founded_by])

##
matches = matcher(doc1)
depmatches = dependency_matcher(doc1)

results = []
for match_id, start, end in matches:
    matched_tokens = doc1[start:end+1]
    token_texts = [token.text for token in matched_tokens]

    # Append results to the list with mapped data
    results.append({
        "Match_ID": matcher.vocab.strings[match_id],
        "Matched_Tokens": " ".join(token_texts),
        "Start_Index": start,
        "End_Index": end
    })

for match_id, indexes in depmatches:
    matched_tokens = [doc1[i] for i in indexes]
    token_texts = [token.text for token in matched_tokens]

    if matcher.vocab.strings[match_id] == "OBJECTIVE":
        results.append({
            "Match_ID": matcher.vocab.strings[match_id],
            "Matched_Tokens": " ".join(token_texts),
            "Start_Index": indexes[0],
            "End_Index": indexes[len(indexes) - 2]
        })

    else:
        results.append({
            "Match_ID": matcher.vocab.strings[match_id],
            "Matched_Tokens": " ".join(token_texts),
            "Start_Index": indexes[len(indexes)-2],
            "End_Index": indexes[len(indexes)-1]
        })

df_matches = pd.DataFrame(results)

df['match_id'] = None

for idx, row in df_matches.iterrows():
    # Assign the match tag to the corresponding range of indexes
    df.loc[row['Start_Index']:row['End_Index']-1, 'match_id'] = row['Match_ID']


## ------------------------------------------------------------------------------
#CREATE A DF WITH THE SENTENCES AND MATCHES

data = []

for sent in doc1.sents:
    sentence_text = sent.text
    iob_tags = []
    for token in sent:
        if token.ent_iob_ == "O":
            iob_tags.append(token.ent_iob_)
        else:
            iob_tags.append(token.ent_iob_ + "-" + token.ent_type_)
    iob_tags_str = ' '.join(iob_tags)
    data.append({"sentence": sentence_text, "IOB_tag": iob_tags_str})

df1 = pd.DataFrame(data)


start_index = []
end_index = []
start = 0
for sent in doc1.sents:
    length_row = len(sent)
    start_index.append(start)
    end_index.append(start+length_row-1)
    start += length_row

df1['start_index'] = start_index
df1['end_index'] = end_index
df1['match_id'] = None

for index, row in df1.iterrows():
        for match_id, indexes in depmatches:
            match_start = min(indexes)
            match_end = max(indexes)
            if row['start_index'] <= match_start < row['end_index'] or row['start_index'] < match_end <= row['end_index']:
                match_name = matcher.vocab.strings[match_id]
                df1.at[index, 'match_id'] = match_name
                break

for index, row in df1.iterrows():
        for match_id, match_start, match_end in matches:
            if row['start_index'] <= match_start < row['end_index'] or row['start_index'] < match_end <= row['end_index']:
                match_name = matcher.vocab.strings[match_id]
                df1.at[index, 'match_id']=match_name
                break

df1['match_id'] = df1['match_id'].apply(lambda x: x if x and len(x) > 0 else '_NO_MATCH_')

## ------------------------------------------------------------------------------
#DATA TOKENIZATION

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df1['sentence'].values)

sequences = tokenizer.texts_to_sequences(df1['sentence'].values)

IOB_tags = df['tag'].unique()
# Map each tag to an integer
tag_encoder = LabelEncoder()
tag_encoder.fit(IOB_tags)
tag_to_int = {tag: idx + 1 for idx, tag in enumerate(tag_encoder.classes_)}

df1['IOB_tag_sequences'] = [[tag_to_int[tag] for tag in seq.split()] for seq in df1['IOB_tag']]

vocabulary_size_sentence = len(tokenizer.word_counts)

## ------------------------------------------------------------------------------
#PADDING
sequence_len = np.array([len(s) for s in sequences])
longest_sequence = sequence_len.max() #73 tokens in the longest sequence

print([(str(p) + '%', np.percentile(sequence_len, p)) for p in range(75,101, 5)])
max_sequence_len = 73

X = pad_sequences(sequences, maxlen=max_sequence_len)
X_iob = pad_sequences(df1['IOB_tag_sequences'], maxlen=max_sequence_len)

label_encoder = LabelEncoder()
df1['match_id_encoded'] = label_encoder.fit_transform(df1['match_id'])

Y = to_categorical(df1['match_id_encoded'])

match_id_to_class = {idx: label for idx, label in enumerate(label_encoder.classes_)}
class_to_match_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}

tokenizer.index_word[0] = '_PAD_'

## ------------------------------------------------------------------------------
# TEST AND TRAIN SPLIT

X_train_sentences, X_test_sentences, X_train_tags, X_test_tags, Y_train_matches, Y_test_matches = train_test_split(
X, X_iob, Y, test_size=0.25, random_state=42)

print(X_train_sentences.shape, X_train_tags.shape, Y_train_matches.shape)
print(X_test_sentences.shape, X_test_tags.shape, Y_test_matches.shape)

## ------------------------------------------------------------------------------
# LOADING GLOVE

def load_glove_embedding_matrix(word_index, embed_dim):
    """Load Glove embeddings."""

   # !wget http://nlp.stanford.edu/data/glove.6B.zip     ##use this code on colab
    # !unzip glove*.zip    ##use this code on colab

    glove_zip_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_zip_path = "glove.6B.zip"
    glove_dir = "glove_files"

    # Step 1: Download the GloVe zip file if not already downloaded. NOT NECESSARY ON COLAB
    if not os.path.exists(glove_zip_path):
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(glove_zip_url, glove_zip_path)
        print("Download complete.")

        # Step 2: Extract the GloVe files if not already extracted   NOT NECESSARY ON COLAB
    if not os.path.exists(glove_dir):
        print("Extracting GloVe files...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zf:
            zf.extractall(glove_dir)
        print("Extraction complete.")

        # Step 3: Load the desired GloVe file
    glove_file_path = os.path.join(glove_dir, f"glove.6B.{embed_dim}d.txt")
    if not os.path.exists(glove_file_path):
        raise FileNotFoundError(f"GloVe file for {embed_dim} dimensions not found.")

     # path = 'glove.6B.100d.txt'

    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


USE_GLOVE=True
glove_matrix=None
if USE_GLOVE:
    embedding_dim = 100
    glove_matrix = load_glove_embedding_matrix(tokenizer.word_index, embedding_dim)

## ------------------------------------------------------------------------------
# COMPOSE THE LSTM NETWORK (IT WORKS + PADDING IS MASKED)

#Note that embed_dim, lstm_out, batch_size, droupout_x variables are hyperparameters, their values are somehow
# intuitive, can be and must be played with in order to achieve good results
#note that I am using softmax as activation function. The reason is that our Network is using categorical crossentropy,
# and softmax is just the right activation method for that.

embed_dim = 100
lstm_out = 150

# Sentence Input and Embedding
sentence_input = Input(shape=(73, ), name="sentence_input")
sentence_embedding = Embedding(vocabulary_size_sentence+1, embed_dim,
                                weights=[glove_matrix],
                                trainable=True)(sentence_input)

# IOB Tag Input and Embedding
iob_input = Input(shape=(73, ), name="iob_input")
iob_embedding = Embedding(vocabulary_size_sentence+1, embed_dim,
                           trainable=True)(iob_input)


combined = concatenate([sentence_embedding, iob_embedding])
masked_combined = Masking(mask_value=0.0)(combined)

lstm_output = LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(masked_combined)

output = Dense(Y.shape[1], activation="softmax")(lstm_output)

model = Model(inputs=[sentence_input, iob_input], outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

## TRAIN THE NETWORK

best_model_file = 'lstm-conll-best-model.weights.h5'

checkpoint = ModelCheckpoint(
    best_model_file,
    save_weights_only=True,
    save_best_only=True
)

early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto",
                                        restore_best_weights=True)

model.fit(x=[X_train_sentences, X_train_tags], y = Y_train_matches,
          validation_data=([X_test_sentences, X_test_tags], Y_test_matches),
          epochs = 90, verbose = 2, batch_size=32)

prediction_probas = model.predict([X_test_sentences, X_test_tags])
predictions = [np.argmax(pred) for pred in prediction_probas]
score,acc = model.evaluate([X_test_sentences, X_test_tags], Y_test_matches, verbose = 2, batch_size = 128)

print("acc: %.2f" % (acc))


## ------------------------------------------------------------------------------
# CONFUSION MATRIX

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


prediction_probas = model.predict([X_test_sentences, X_test_tags])
predictions = np.argmax(prediction_probas, axis=1)
Y_true = np.argmax(Y_test_matches, axis=1)

cm = confusion_matrix(Y_true, predictions, labels=range(9))  # Force computation for all 9 classes

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=[f'{match_id_to_class[i]}' for i in range(9)],
            yticklabels=[f'{match_id_to_class[i]}' for i in range(9)])

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show(block=True)
plt.interactive(False)


## ------------------------------------------------------------------------------
# PREDICTION ON NEW SENTENCE

sentence_prediction = 'Rebels of ISIL attacked Syria'
sentence_prediction = [tokenizer.texts_to_sequences([sentence_prediction])[0]]
sentence_prediction = pad_sequences(sentence_prediction, maxlen=max_sequence_len)
IOB_tags_prediction = 'O O B-Organisation O B-Location'
IOB_tags_prediction = [[tag_to_int[tag] for tag in IOB_tags_prediction.split()]]
IOB_tags_prediction = pad_sequences(IOB_tags_prediction, maxlen=max_sequence_len)

print(sentence_prediction)
print(IOB_tags_prediction)

match_ID_prediction = model.predict([sentence_prediction, IOB_tags_prediction],batch_size=1,verbose = 2)[0]
predicted_value = np.argmax(match_ID_prediction)
print(match_id_to_class[predicted_value]) #LOCATION_OF_OPERATION

## ------------------------------------------------------------------------------
# VALIDATION ON SECOND FILE


