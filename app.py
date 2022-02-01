import streamlit as st
import pickle
import sklearn
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps  = PorterStemmer()

def text_cleaner(text):
    text = text.lower()                        # lower case ('HELLO' --> 'hello')
    text = nltk.word_tokenize(text)            # tokenization (['hello', 'how', 'are', 'you'])
    
    y = [] 
    for i in text:                             #removing ".,?!"
        if i.isalnum():
            y.append(i)   
    text = y.copy()
    y.clear()
                                              # removing stopwords('is,are,it) and punctuations($,%,><)
    for i in text:                              
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y.copy()
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)   


tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title('Messages Classifier')

input_sms = st.text_area("Enter the message")

if st.button('Pridict'):

    text_cleaner = text_cleaner(input_sms)

    vector_input = tf.transform([text_cleaner])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')