import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.layers import Input, Dense
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import format_db
from keras.utils.vis_utils import plot_model

tf.random.set_seed(0)

format_db.format()

df_users = pd.read_csv('db/users_formatted.csv')
df_items = pd.read_csv('db/items_formatted.csv')
df_interactions = pd.read_csv('db/interactions.csv')

df_interactions = pd.merge(df_interactions, df_users.rename(columns={'id': 'userID'}), on='userID')
df_interactions = pd.merge(df_interactions, df_items.rename(columns={'id': 'itemID'}), on='itemID')
df_interactions.drop('userID', axis=1, inplace=True)
df_interactions.drop('itemID', axis=1, inplace=True)

y = [int(x) for x in df_interactions['interaction'].values.tolist()]
df_interactions.drop('interaction', axis=1, inplace=True)
X = df_interactions.values.tolist()
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)


n_input_features = len(X[0])
print(f'Number of input fatures: {n_input_features}')

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

np.save('db/train_test/X_train.npy', X_train)
np.save('db/train_test/X_test.npy', X_test)
np.save('db/train_test/y_train.npy', y_train)
np.save('db/train_test/y_test.npy', y_test)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

print(f'Number of positive samples train set: {np.count_nonzero(y_train == True)}')
print(f'Number of positive samples test set: {np.count_nonzero(y_test == True)}')

#Neural Network
l2_reg = 0.001
# l2_reg = 0.0005
model = Sequential([Input(shape=(n_input_features,), name='input'),
Dense(64, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 0),
Dense(32, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 1),
Dense(16, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 2),
# Dense(8, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 3),
Dense(4, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), activation='relu', name='layer%d' % 4),
Dense(1, activation='sigmoid', name='prediction')])

# binary crossentropy loss function with Adam optimizer
learning_rate = 0.02
model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=["accuracy"])

# summarize layers
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plt.show(block=True)

# print(len(X_train))
# print(y_train)
# print(len(y_train))

history = model.fit(X_train, y_train, batch_size=32, epochs=50,
                    validation_data=(X_test, y_test))
model.save('model/seq.keras')
score = model.evaluate(X_test, y_test, verbose=1)
print(score)

print('Model evaluation')
test_scores = model.predict(X_test)
print(test_scores)
test_preds = (test_scores >= 0.25)
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

y_pred_keras = test_scores.ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show(block=True)

