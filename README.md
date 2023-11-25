# NLP
DA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis")
train_data = dataset["train"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    train_data["content"], train_data["polarity"], test_size=0.2, random_state=42
)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Classifier Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
