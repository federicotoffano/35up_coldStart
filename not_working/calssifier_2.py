import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Dense
from keras.layers.merging import concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import random as rnd
import tensorflow as tf


tf.random.set_seed(1)
rnd.seed(1)

df_users = pd.read_csv('db/users_ml.csv')
df_movies = pd.read_csv('db/items_ml.csv')
df_interactions = pd.read_csv('../db/interactions.csv')

df_interactions_users = pd.merge(df_interactions, df_users.rename(columns={'id': 'userID'}), on='userID')
df_interactions_users.drop('userID', axis=1, inplace=True)
df_interactions_users.drop('itemID', axis=1, inplace=True)
#95cols
df_interactions_items = pd.merge(df_interactions, df_movies.rename(columns={'id': 'itemID'}), on='itemID')
df_interactions_items.drop('userID', axis=1, inplace=True)
df_interactions_items.drop('itemID', axis=1, inplace=True)


#i.e., last 60 columns dscribe movie

# print(df_interactions)
# print(df_interactions.columns)


# df_interactions.to_csv('db/merge_ml.csv', index=False)

# df_interactions_pos = df_interactions[df_interactions['interaction']==True]
# df_interactions_neg = df_interactions[df_interactions['interaction']==False]

# X_pos = df_interactions_pos[df_interactions_pos.columns[1:]].values.tolist()
# y_pos = [int(x) for x in df_interactions_pos['interaction'].values.tolist()]
# X_neg = df_interactions_neg[df_interactions_neg.columns[1:]].values.tolist()
# y_neg = [int(x) for x in df_interactions_neg['interaction'].values.tolist()]
# X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)
# X_train = np.concatenate((np.array(X_train_pos), np.array(X_neg)))
# X_test = np.array(X_test_pos)
# y_train = np.concatenate((np.array(y_train_pos), np.array(y_neg)))
# y_test = np.array(y_test_pos)


X_users = df_interactions_users[df_interactions_users.columns[1:]].values.tolist()
X_items = df_interactions_items[df_interactions_items.columns[1:]].values.tolist()
y = [int(x) for x in df_interactions_users['interaction'].values.tolist()]
X_train_users, X_test_users, X_train_items, X_test_items, y_train, y_test = \
    train_test_split(X_users, X_items, y, test_size=0.33, random_state=42)

n_user_features = len(X_users[0])
print(n_user_features)
n_movie_features = len(X_items[0])
print(n_movie_features)

X_train_users = np.array(X_train_users)
X_test_users = np.array(X_test_users)
X_train_items = np.array(X_train_items)
X_test_items = np.array(X_test_items)
y_train = np.array(y_train)
y_test = np.array(y_test)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
#
print(X_train_users.shape)
print(X_test_users.shape)
print(X_train_items.shape)
print(X_test_items.shape)
print(y_train.shape)
print(y_test.shape)

print(np.count_nonzero(y_test == True))
print(np.count_nonzero(y_train == True))

l2_reg = 0.0001

num_outputs = 8
user_net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu'),
    tf.keras.layers.Dense(num_outputs)
])

movie_net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu'),
    tf.keras.layers.Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu'),
    tf.keras.layers.Dense(num_outputs)
])

input_user = tf.keras.layers.Input(shape=(n_user_features))
user_vec = user_net(input_user)
user_vec = tf.linalg.l2_normalize(user_vec, axis=1)

input_movie = tf.keras.layers.Input(shape=(n_movie_features))
movie_vec = movie_net(input_movie)
movie_vec = tf.linalg.l2_normalize(movie_vec, axis=1)

output = tf.keras.layers.Dot(axes=1)([user_vec, movie_vec])

model = tf.keras.Model([input_user, input_movie], output)

model.summary()

# model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn,
              metrics=["accuracy"])
print(len(X_train_users))
print(y_train)
print(len(y_train))
history = model.fit([X_train_users, X_train_items], y_train, batch_size=32, epochs=100,
                    validation_data=([X_test_users, X_test_items], y_test))
score = model.evaluate([X_test_users, X_test_items], y_test, verbose=1)
print(score)

# train_preds = (model.predict(X_train) >= 0.7)
print(model.predict([X_test_users, X_test_items]))
test_preds = (model.predict([X_test_users, X_test_items]) >= 0.5)

# print(y_test_i)
# print(test_preds)
matrix = metrics.confusion_matrix(y_test, test_preds)
print(matrix)
disp = metrics.ConfusionMatrixDisplay(matrix)
disp.plot()
plt.show()

print('scores')
print(f1_score(y_test, test_preds, average="macro"))
print(precision_score(y_test, test_preds, average="macro"))
print(recall_score(y_test, test_preds, average="macro"))
print('-----')