from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
#Task 1
data_age = ['youth', 'youth', 'middle_aged', 'senior', 'middle_aged', 'senior', 'youth', 'middle_aged', 'middle_aged', 'senior']
data_income = ['high', 'high', 'high', 'medium', 'low', 'low', 'medium', 'medium', 'high', 'medium']
data_student = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes']
data_credit_rating = ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'excellent']
data_buys_computer = ['no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes']
le = preprocessing.LabelEncoder()
age_encoded = le.fit_transform(data_age)
income_encoded = le.fit_transform(data_income)
student_encoded = le.fit_transform(data_student)
credit_rating_encoded = le.fit_transform(data_credit_rating)
buys_computer_encoded = le.fit_transform(data_buys_computer)
features = list(zip(age_encoded, income_encoded, student_encoded, credit_rating_encoded))
labels = buys_computer_encoded
features_train, features_test, label_train, label_test = train_test_split(features, labels, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(features_train, label_train)
predicted = model.predict(features_test)
print("Predictions:", predicted)
conf_mat = confusion_matrix(label_test, predicted)
print("Confusion Matrix:")
print(conf_mat)
accuracy = accuracy_score(label_test, predicted)
print("Accuracy:", accuracy)


# Task 2
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
weather = ['sunny', 'sunny', 'rainy', 'sunny', 'rainy', 'rainy', 'sunny']
temperature = ['hot', 'hot', 'cold', 'cold', 'cold', 'hot', 'hot']
play = ['yes', 'yes', 'no', 'yes', 'no', 'no', 'yes']
le = preprocessing.LabelEncoder()
weather_encoded = le.fit_transform(weather)
temperature_encoded = le.fit_transform(temperature)
play_encoded = le.fit_transform(play)
features = list(zip(weather_encoded, temperature_encoded))
features_train, features_test, label_train, label_test = train_test_split(features, play_encoded, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(features_train, label_train)
predicted = model.predict(features_test)
print("Predictions:", predicted)
conf_mat = confusion_matrix(label_test, predicted)
print("Confusion Matrix:")
print(conf_mat)
accuracy = accuracy_score(label_test, predicted)
print("Accuracy:", accuracy)
accuracy_manual = (2 + 2) / (2 + 2 + 0 + 0)
print("Calculated Accuracy:", accuracy_manual)