from sklearn.svm import SVC
from lab1_task5 import f1_score
from lab1_task5 import precision_score
from lab1_task5 import recall_score
from lab1_task5 import accuracy_score
from lab1_task5 import df

X = df.drop(columns=['actual_label'])
y = df['actual_label']

svm_model = SVC()

svm_model.fit(X, y)

y_pred_svm = svm_model.predict(X)

accuracy_svm = accuracy_score(y, y_pred_svm)

recall_svm = recall_score(y, y_pred_svm)

precision_svm = precision_score(y, y_pred_svm)

f1_svm = f1_score(y, y_pred_svm)

print('Accuracy SVM:', accuracy_svm)
print('Recall SVM:', recall_svm)
print('Precision SVM:', precision_svm)
print('F1 Score SVM:', f1_svm)
