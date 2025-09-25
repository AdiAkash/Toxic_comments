import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ========================
# Step 1: Create dataset directly inside Python
# ========================
data = {
    "comment_text": [
        "You are so stupid!",
        "I love this product amazing quality.",
        "What a horrible experience I hate it!",
        "This is so cool thanks for sharing.",
        "You are a disgusting person.",
        "Such a wonderful day feeling blessed!",
        "I will kill you",
        "Thank you for your help I appreciate it.",
        "Worst service ever never coming back.",
        "This place is fantastic great food and staff."
    ],
    "toxic": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# ========================
# Step 2: Train/test split
# ========================
X = df["comment_text"]
y = df["toxic"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========================
# Step 3: Build Pipeline
# ========================
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)

# ========================
# Step 4: Test model
# ========================
print("Model Accuracy on test set:", model.score(X_test, y_test))

# ========================
# Step 5: Try your own comment
# ========================
while True:
    user_input = input("\nType a comment (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    prediction = model.predict([user_input])[0]
    if prediction == 1:
        print("⚠️ Toxic Comment Detected")
    else:
        print("✅ Non-toxic Comment")
