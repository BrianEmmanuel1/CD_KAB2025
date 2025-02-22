from dataset_preprocessing import split_cifar_data
import decision_tree as dt

x_train, y_train, x_test, y_test = split_cifar_data()
model = dt.classification_tree(x_train=x_train, y_train=y_train)
y_pred = dt.predict(model, x_test)
ac = dt.accuracy_class(y_test, y_pred)
print(f"Accuracy: {ac}")