from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Gather training data from the user
print("Spam Trainer")
print("Enter email samples and tag them.")
print("Type 'done' when you're finished.\n")

emails = []
labels = []

while True:
    email = input("Email: ")
    if email.lower() == 'done':
        break
    label = input("Spam? (yes/no): ").strip().lower()
    while label not in ['yes', 'no']:
        label = input("Please enter 'yes' or 'no': ").strip().lower()
    
    emails.append(email)
    labels.append(1 if label == 'yes' else 0)

# Step 2: Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Step 3: Train the classifier
model = MultinomialNB()
model.fit(X, labels)

# Step 4: Test phase
print("\n--- Test Mode ---")
while True:
    test_email = input("Test email (or type 'quit'): ")
    if test_email.lower() == 'quit':
        break
    test_vector = vectorizer.transform([test_email])
    prediction = model.predict(test_vector)[0]
    print("Spam" if prediction == 1 else "Not Spam")
