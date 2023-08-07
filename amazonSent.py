import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# https://www.kaggle.com/datasets/magdawjcicka/amazon-reviews-2018-electronics

data = pd.read_csv('C:/Users/mourt/Amazon/AmazonReviews.csv')

print("----Initial dataframe----")
head = data.head(3)
print(head.to_string(index=False))


data.info()
data.dropna(inplace=True)

#negative -> 1,2,3
#positive -> 4,5
data.loc[data['Rating']<=3,'Rating'] = 0
data.loc[data['Rating']>3,'Rating'] = 1

def clean_reviews(review):
    stop_words = stopwords.words('english')
    clean_review = " ".join(word for word in review.split() if word not in stop_words)
    return clean_review

data['Review'] = data['Review'].apply(clean_reviews)

print("----Cleaned dataframe----")
data.head()
cleaned_head = data.head(3)
print(cleaned_head.to_string(index=False))

data['Rating'].value_counts()
print("Amount of negative (0) and positive (1) reviews")
print(data['Rating'].value_counts())



consolidated=' '.join(word for word in data['Review'][data['Rating']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(25,20))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()

consolidated=' '.join(word for word in data['Review'][data['Rating']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(25,20))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()

vectorizer = TfidfVectorizer(max_features=2500)
features = vectorizer.fit_transform(data['Review'] ).toarray()

feature_train ,feature_test,sentiment_train,sentiment_test = train_test_split(features,data['Rating'], test_size=0.25 , random_state=42)

model=LogisticRegression()
model.fit(feature_train,sentiment_train)
 
#testing the model
prediction = model.predict(feature_test)
 
#model accuracy
print("--Accuracy score--")
print(accuracy_score(sentiment_test,prediction))

print("--Precision score--")
print(precision_score(sentiment_test,prediction))

print("--Recall score--")
print(recall_score(sentiment_test,prediction))

cm = confusion_matrix(sentiment_test,prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True]) 
cm_display.plot()
plt.show()






