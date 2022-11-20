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
# from tensorflow.linalg import l2_normalize
import tensorflow as tf





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



l2_reg = 0.001

# #user net
# user_input = Input(shape=(n_user_vars,), name='user_input')
# user_dense1 = Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
#                     activation='relu', name='layer_user%d' % 1)(user_input)
# # user_dense2 = Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
# #                     activation='relu', name='layer_user%d' % 2)(user_dense1)
#
# #item net
# item_input = Input(shape=(n_movie_vars,), name='item_input')
# item_dense1 = Dense(32, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
#                     activation='relu', name='layer_item%d' % 1)(item_input)
# item_dense2 = Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
#                     activation='relu', name='layer_item%d' % 2)(item_dense1)
# item_dense3 = Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
#                     activation='relu', name='layer_item%d' % 3)(item_dense2)

#merge user and item

num_outputs = 8
user_net = Sequential([
    Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu'),
    Dense(num_outputs)
])

movie_net = Sequential([
    Dense(32, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu'),
    Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu'),
    Dense(num_outputs)
])

input_user = Input(shape=(n_user_features))
user_vec = user_net(input_user)
user_vec = tf.linalg.l2_normalize(user_vec, axis=1)

input_movie = Input(shape=(n_movie_features))
movie_vec = movie_net(input_movie)
movie_vec = tf.linalg.l2_normalize(movie_vec, axis=1)


merge = concatenate([user_vec, movie_vec], axis=1)

#classification net
common_dense1 = Dense(8, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
      activation='relu', name='layer_common%d' % 1)(merge)
common_dense2 = Dense(4, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
      activation='relu', name='layer_common%d' % 2)(common_dense1)
# common_dense3 = Dense(4, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg),
#       activation='relu', name='layer_common%d' % 3)(common_dense2)
output = Dense(1, activation='sigmoid', name='prediction')(common_dense2)
model = Model(inputs=[input_user, input_movie], outputs=output)
# summarize layers
print(model.summary())
# plot graph
# plot_model(model, to_file='shared_input_layer.png')

 # Final prediction layer
# model = Sequential([Input(shape=(len(X_train[0]),), dtype='int32', name='user_input'),
# Dense(32, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 1),
# Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 2),
# Dense(8, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 3),
# Dense(4, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 4),
# Dense(1, activation='sigmoid', name='prediction')])

# model = Sequential(input=input,
#               output=prediction)
learning_rate = 0.05
model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

model.summary()

# model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
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