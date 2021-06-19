import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import twitter_new
import nltk
from nltk.tokenize import PunktSentenceTokenizer

from nltk.corpus import state_union
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def remove_pattern(input_txt, pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
    #print(input_txt)    
    return input_txt 


   

def preprocess_string(str_arg):
    
    """"
        Parameters:
        ----------
        str_arg: example string to be preprocessed
        
        What the function does?
        -----------------------
        Preprocess the string argument - str_arg - such that :
        1. everything apart from letters is excluded
        2. multiple spaces are replaced by single space
        3. str_arg is converted to lower case 
        
        Example:
        --------
        Input :  Menu is absolutely perfect,loved it!
        Output:  menu is absolutely perfect loved it
        
        Returns:
        ---------
        Preprocessed string 
        
    """
    cleaned_str=""
    clean_tweet = re.match('(.?)http.?\s?(.*?)', str_arg)
    #cleaned_str=np.vectorize(remove_pattern)( np.vectorize(remove_pattern)(str_arg,r'http.?://[^\s]+[\s]?'),r'https.?://[^\s]+[\s]?')
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case

    #print(cleaned_str)
    # sample_text = cleaned_str
    
    # tokenized = custom_sent_tokenizer.tokenize(sample_text)
    return process_content(cleaned_str)

    #return cleaned_str
 #    lemmatizer = WordNetLemmatizer()
 #    ps = PorterStemmer()
 #    stop_words = set(stopwords.words("english"))

 #    words = word_tokenize(str(cleaned_str))  
 #    filtered_sent = [w for w in words if not w in stop_words]
 #    filtered_sent2 = []
 #    for j in filtered_sent:
 #    	filtered_sent2.append(lemmatizer.lemmatize(j)) 

	# return filtered_sent2 # returning the preprocessed string 
def process_content(c_str):
    try:
        
        words = nltk.word_tokenize(c_str)
        tagged = nltk.pos_tag(words)
        sig_str=""
        for i in tagged:
            if(i[1]=="RBR" or i[1]=="RBS" or i[1]=="RB" or i[1]=="JJ" or i[1]=="JJR" or i[1]=="JJS"):
                sig_str=sig_str+" "+i[0]+" "
        return sig_str

        #print(tagged)

        

    except Exception as e:
        print(str(e))

df_tweet = pd.read_csv('Test.csv')
my_list = df_tweet['Tweets'].tolist()
response= df_tweet['Response'].tolist()
# print(my_list)
train_data= my_list
train_labels= response  
#train_text = state_union.raw("2005-GWBush.txt")
#custom_sent_tokenizer = PunktSentenceTokenizer(train_text) 
train_data=[preprocess_string(train_str) for train_str in train_data]
print(train_data)
print ("Data Cleaning Done")

# df_tweet = pd.read_csv('tweet_Verizon.csv')
# my_list = df_tweet['0'].tolist()
# response= df_tweet['Response'].tolist()
# # print(my_list)
# train_data= my_list
# train_labels= response   
# train_data=[preprocess_string(train_str) for train_str in train_data]
# print ("Data Cleaning Done")
# print ("Total Number of Training Examples: ",len(train_data))
# print(train_data[1])

count_vect = CountVectorizer() #instantiate it's object
X_train_counts = count_vect.fit_transform(train_data) #builds a term-document matrix ands return it
print (X_train_counts.shape)

clf = MultinomialNB() #simply instantiate a Multinomial Naive Bayes object
clf.fit(X_train_counts, train_labels)  #calling the fit method trains it
print ("Training Completed")

# test_data=newsgroups_test.data #get test set examples
# test_labels=newsgroups_test.target #get test set labels

# print ("Number of Test Examples: ",len(test_data))
# print ("Number of Test Labels: ",len(test_labels))

# test_data=[preprocess_string(test_str) for test_str in test_data] #need to preporcess the test set as well!!
# print ("Number of Test Examples: ",len(test_data))

# X_test_counts=count_vect.transform(test_data) #transforms test data to numerical form
# print (X_test_counts.shape)

# predicted=clf.predict(X_test_counts) #simply call the predict function to predict for test set
# print ("Test Set Accuracy : ",np.sum(predicted==test_labels)/float(len(predicted)))

# Next task : remove the required parts of speech from tagged/cleaned_str


df_tweet_test = pd.read_csv('tweet_Verizon2.csv')
my_list_test = df_tweet_test['Tweets'].tolist()
response_test= df_tweet_test['Response'].tolist()

test_data=my_list_test #get test set examples
test_labels=response_test

test_data=[preprocess_string(test_str) for test_str in test_data] #need to preporcess the test set as well!!
print ("Number of Test Examples: ",len(test_data))


X_test_counts=count_vect.transform(test_data) #transforms test data to numerical form
print (X_test_counts.shape)

predicted=clf.predict(X_test_counts)
print(predicted) #simply call the predict function to predict for test set
print ("Test Set Accuracy : ",np.sum(predicted==test_labels)/float(len(predicted)))