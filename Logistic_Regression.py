import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def extract_features(field, training_data, testing_data):
    cv = CountVectorizer(binary=False)   # no need df.max_df because I removed the stopwords
    cv.fit_transform(training_data[field].values.astype('U'))  # learning about the text
    train_feature_set = cv.transform(training_data[field].values.astype('U'))
    test_feature_set = cv.transform(testing_data[field].values.astype('U'))
    return train_feature_set, test_feature_set, cv

# Read dataset
df = pd.read_csv('clean_dataset.csv')
avg = 0
n = 100
for i in range(n):
    df = sklearn.utils.shuffle(df)

    # Split to train and set
    training_data, testing_data = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)

    # Labels
    Y_train = training_data['class'].values
    Y_test = testing_data['class'].values

    # Extract features
    X_train, X_test, feature_transformer = extract_features('text', training_data, testing_data)

    # Logistic Regression
    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', random_state=0, C=5, penalty='l2', max_iter=1000)
    model = scikit_log_reg.fit(X_train, Y_train)  # learning with the features we fount and the train set

    ytest = np.array(Y_test)
    print("Result of Logistic Regression Model...")
    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    cm = confusion_matrix(ytest, model.predict(X_test))
    print(cm)
    avg += accuracy_score(Y_test, model.predict(X_test))
print("average accuracy score:  ", avg / n)
df.groupby('class').text.count().plot.bar(ylim=0)

plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Suicide', 'Predicted non-Suicide'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Suicide', 'Actual non-Suicide'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title('Logistic Regression average score=%s ' % (avg / n))
plt.show()
