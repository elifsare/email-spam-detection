# Loading the data
import pandas as pd
rawData = pd.read_csv('D:/elif/calisma_alani/mail_data.csv')
print(rawData)

# replace the null values a null string
mailData = rawData.where((pd.notnull(rawData)), '') 
print(mailData.head())

#checking the number of rows and colums
print("satır, sütun --> " + str(mailData.shape))

# label spam mail as 0; ham mail as 1;
mailData.loc[mailData['Category'] == 'spam', 'Category',] = 0
mailData.loc[mailData['Category'] == 'ham', 'Category',] = 1

# separating the data as texts and label
X = mailData['Message']
y = mailData['Category']
print(X)
print(y)

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 3)  

# Feature Extraction
# transform the text data to feature vectors that can be used as input to the ligistic regression
from sklearn.feature_extraction.text import TfidfVectorizer 
feature_extraction = TfidfVectorizer(min_df = 1,
                                     stop_words = 'english',
                                     lowercase = 'True')

# X has messages --> so we will apply TfidfVectorizer to it
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
print("X_train_features: \n {} \n X_test_features: \n {}".format(X_train_features, X_test_features))

# convert object type to integer
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Training model
# prediction on training data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train_features, y_train)

#evuluating the trained model
y_train_pred = model.predict(X_train_features)

from sklearn.metrics import accuracy_score
train_accuracy_score = accuracy_score(y_train, y_train_pred)
print("train_accuracy_score: {}".format(train_accuracy_score))

# prediction on training data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train_features, y_train)

#evuluating the test data
y_test_pred = model.predict(X_test_features)

test_accuracy_score = accuracy_score(y_test, y_test_pred)
print("test_accuracy_score: {}".format(test_accuracy_score))
#The success value in the test set indicates that the model is not overfitting.

# Let's choose a mail from the data file and test it in the model
input_m = ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
input_features = feature_extraction.transform(input_m)
pred = model.predict(input_features)
if(pred[0] == 1):
    print('ham mail')

else:
    print('spam mail')
