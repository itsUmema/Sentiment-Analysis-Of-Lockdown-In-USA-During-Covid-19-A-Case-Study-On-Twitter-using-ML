"""
****************************************************************************
Developed by 
            ---> TEAM OCTATES <---
                  KAHAKASHAN
                  UMEMA ZAIB
                  ABDUL REHMAN BELGAUMI
                  IFRAH FARHEEN
****************************************************************************

"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X_train = ["I'm terrified of COVID-19 and its impact on our society",
    "I'm frustrated with the way the government is handling the pandemic",
    "I appreciate the efforts of healthcare workers during these tough times",
    "The vaccine rollout has been impressive, and I'm hopeful for the future",
    "COVID-19 has disrupted my life in so many ways, and I'm struggling to cope",
    "I'm grateful for the opportunity to work from home and stay safe",
    "The misinformation about COVID-19 is causing unnecessary panic",
    "I lost a loved one to COVID-19, and it's been devastating",
    "Wearing masks and social distancing is a small sacrifice to protect others",
    "I'm tired of hearing about COVID-19 in the news every day", "The environment and nature is healing because of covid"]

y_train = ["negative","negative","positive","positive","negative","positive","negative","negative","positive","negative","positive"]
#nltk.download('stopwords')
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
s1=input("***ENTER THE KEYWORD***\n")
def getCleanedText(text):
  text = text.lower()

  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text

X_test = [s1]
X_clean = [getCleanedText(i) for i in X_train]
xt_clean = [getCleanedText(i) for i in X_test]
cv = CountVectorizer(ngram_range = (1,2))
X_vec = cv.fit_transform(X_clean).toarray()
Xt_vect = cv.transform(xt_clean).toarray()
mn = MultinomialNB()
mn.fit(X_vec, y_train)
y_pred = mn.predict(Xt_vect)
print(*y_pred)