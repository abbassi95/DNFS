import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import requests
import tarfile
from sklearn.preprocessing import LabelEncoder
import numpy as np
from feature_selector import FeatureSelector
from tensorflow.keras.utils import to_categorical

#%% URL of the dataset
url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"

#%%Download the tar.gz file
response = requests.get(url)
open("20news-bydate.tar.gz", "wb").write(response.content)

#%%Extract the tar.gz file
with tarfile.open("20news-bydate.tar.gz", "r:gz") as tar:
    tar.extractall()

#%%Function to read the dataset into a DataFrame
def read_newsgroups_data(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                category = root.split(os.sep)[-1]
                data.append((category, content))
    df = pd.DataFrame(data, columns=['Category', 'Content'])
    return df

#%% Reading the dataset
df_train = read_newsgroups_data("20news-bydate-train")
df_test = read_newsgroups_data("20news-bydate-test")
df = pd.concat([df_train, df_test], ignore_index=True)

#%% Filter to keep only two categories
selected_classes = ['sci.space', 'rec.autos']
df_binary = df[df['Category'].isin(selected_classes)].copy()

#%% Prepare the data
X = df_binary['Content']
y = df_binary['Category']

#%% Convert text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
X_tfidf = tfidf.fit_transform(X)

#%% Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

X_train = X_train.toarray()
X_test = X_test.toarray()

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#%%Define the Keras model
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

selector = FeatureSelector(task_nodes = [X_train.shape[1], 512, 256, 20],
               selector_nodes = [X_train.shape[1],512,256,512, X_train.shape[1]],
               task_activation='sigmoid')

selector.compile(factor_loss=.5,
            regression = False,
            loss = 'categorical_crossentropy',
            metric = 'categorical_accuracy',
            threshold=0.01) #%%For non-linear classification change threshold to 0.1

# history = selector.fit(100, X_train, y_train_encoded, validation_data=(X_test, y_test_encoded))

#%% Print feature selection. Not that just the feature indices are printed. 

selector.last_epoch = 83

selected_features = selector.eval()
print(f'Selected: {selected_features}. Size: {len(selected_features)}')

#%% Evaluate with SVM
from sklearn.metrics import accuracy_score
import numpy as np

# Build and train SVM on all features
selector.build_SVM(is_regression=False, kernel="rbf")
selector.SVM.fit(X_train[:], y_train_encoded[:])

# Predict class labels
y_train_pred = selector.SVM.predict(X_train)
y_test_pred = selector.SVM.predict(X_test)

# Compute accuracy
acc_train = accuracy_score(y_train_encoded, y_train_pred)
acc_test = accuracy_score(y_test_encoded, y_test_pred)

print(f"Accuracy (train): {acc_train:.4f}")
print(f"Accuracy (test): {acc_test:.4f}")
