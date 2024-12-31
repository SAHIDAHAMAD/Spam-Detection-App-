import pickle
import pandas as pd
df = pd.read_csv(r"C:\D-drive\Datascience notes\Notes\22. Natural Language Processing\SMSSpamCollection",sep='\t',names=['label','message'])
df.head()
df.info()
df.groupby('label').describe()

# Text Cleaning
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(len(df)):
    rp = re.sub('[^a-zA-Z]'," ",df['message'][i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if not word in set(stopwords.words('english'))]
    rp = " ".join(rp)
    corpus.append(rp)

len(corpus)

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
X

y = pd.get_dummies(df['label'],drop_first=True).astype(int)
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Navie Bayes Classifier
#import
from sklearn.naive_bayes import MultinomialNB
# save
model = MultinomialNB()

# fit
model.fit(X_train,y_train)

#Predictions
ypred_test = model.predict(X_test)
ypred_train = model.predict(X_train)

#Evalution
from sklearn.metrics import accuracy_score
print("Train Accuracy:",accuracy_score(y_train,ypred_train))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))

input_mail ='Lottery, you have won a lottery of 1cr'

#convert to dataframe
df_test = pd.DataFrame({'message':input_mail},index=[0])
df_test

# Text Cleaning
corpus = []
for i in range(len(df_test)):
    rp = re.sub('[^a-zA-Z]'," ",df_test['message'][i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if not word in set(stopwords.words('english'))]
    rp = " ".join(rp)
    corpus.append(rp)

print(corpus)

# Text Vectorization
X = cv.transform(corpus).toarray()
X

X.shape

pred = model.predict(X)
pred

if pred==0:
    print('Not Spam')
else:
    print('Spam')

input_mail ='Hi Sahid How are you'
 #convert to dataframe
df_test = pd.DataFrame({'message':input_mail},index=[0])
df_test

# Text Cleaning
corpus = []
for i in range(len(df_test)):
    rp = re.sub('[^a-zA-Z]'," ",df_test['message'][i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if not word in set(stopwords.words('english'))]
    rp = " ".join(rp)
    corpus.append(rp)

# Text Vectorization
X = cv.transform(corpus).toarray()
X

#prediction
pred = model.predict(X)
pred
if pred==0:
    print('Not Spam')
else:
    print('Spam')

# Save the model and vectorizer in a dictionary
filename = 'spam_model.pkl'  # Specify the filename
with open(filename, 'wb') as file:  # Open the file in write-binary mode
    pickle.dump({'model': model, 'vectorizer': cv}, file)  # Save both model and vectorizer

print(f'Model and vectorizer saved to {filename}')


