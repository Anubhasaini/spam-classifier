# ---------------- IMPORT LIBRARIES ----------------
import pandas as pd
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# download stopwords (run once)
nltk.download('stopwords')


# ---------------- LOAD DATA ----------------
data = pd.read_csv("spam.csv", encoding='latin-1')

# taking only useful columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# remove missing values
data.dropna(inplace=True)

# convert text to string
data['message'] = data['message'].astype(str)

# remove blank rows
data = data[data['message'].str.strip() != ""]


# ---------------- LABEL CONVERSION ----------------
# spam = 1, ham = 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# ---------------- TEXT CLEANING FUNCTION ----------------
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)   # remove special chars
    text = text.lower()                    # lowercase
    words = text.split()                   # split words

    # remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    if len(words) == 0:
        return "empty"

    return " ".join(words)


# apply cleaning
data['message'] = data['message'].apply(preprocess)

# remove empty messages
data = data[data['message'] != "empty"]


# ---------------- MODEL BUILDING ----------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message']).toarray()
y = data['label']

model = MultinomialNB()
model.fit(X, y)


# ---------------- STREAMLIT UI ----------------
st.title("📩 Spam Message Classifier")

st.write("Enter a message below and check whether it is spam or not.")

user_msg = st.text_area("Type your message here:")

if st.button("Check Result"):
    if user_msg.strip() == "":
        st.warning("Please enter some text first")
    else:
        cleaned_msg = preprocess(user_msg)

        if cleaned_msg == "empty":
            st.warning("Enter meaningful words")
        else:
            vector_input = vectorizer.transform([cleaned_msg]).toarray()
            prediction = model.predict(vector_input)[0]

            if prediction == 1:
                st.error("🚫 This is a SPAM message")
            else:
                st.success("✅ This is NOT spam")