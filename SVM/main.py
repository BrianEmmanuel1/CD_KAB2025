from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset_preprocessing import preprocess_dataset
# Create a classifier
train_x_features, train_y, test_x_features, test_y = preprocess_dataset()

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(train_x_features, train_y)
print(clf.score(test_x_features, test_y))