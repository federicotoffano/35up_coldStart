from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
import numpy as np

X_test = np.load('db/train_test/X_test.npy')
y_test = np.load('db/train_test/y_test.npy')

model = keras.models.load_model('model/seq.keras')


print('\nModel evaluation...')
score = model.evaluate(X_test, y_test, verbose=1)
test_scores = model.predict(X_test)
test_preds = (test_scores >= 0.2)
matrix = metrics.confusion_matrix(y_test, test_preds)

print('\nConfusion matrix:')
print(matrix)
disp = metrics.ConfusionMatrixDisplay(matrix)
disp.plot()
plt.savefig('confusion_matrix.png')
plt.show()

print()
print(f'F1 score: {f1_score(y_test, test_preds, average="macro")}')
print(f'Precision: {precision_score(y_test, test_preds, average="macro")}')
print(f'Recall: {recall_score(y_test, test_preds, average="macro")}')
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
plt.savefig('ROC.png')
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
plt.savefig('ROC_zoom.png')
plt.show(block=True)




