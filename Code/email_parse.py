# coding: utf-8


from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np

import glob
from email.parser import Parser
parser = Parser()

corpus = []
y = []

for filepath in glob.glob('../Datasets/enron1/ham/*'): 
    with open(filepath, 'r') as ofile:
        email = parser.parse(ofile)
        
        if email.is_multipart():
            for part in email.walk():
                content_type = part.get_content_type()
                content_dispo = str(part.get('Content-Disposition'))
                # skip any text/plain (txt) attachments

                if content_type == 'text/plain' and 'attachment' not in content_dispo:
                    body = part.get_payload(decode=False)
                    break
        else:
            body = email.get_payload(decode=False)

        
        corpus.append(body.decode("ISO-8859-1"))
        y.append(0)

for filepath in glob.glob('../Datasets/enron1/spam/*'): 
    with open(filepath, 'r') as ofile:
        email = parser.parse(ofile)
        
        if email.is_multipart():
            for part in email.walk():
                content_type = part.get_content_type()
                content_dispo = str(part.get('Content-Disposition'))
                # skip any text/plain (txt) attachments

                if content_type == 'text/plain' and 'attachment' not in content_dispo:
                    body = part.get_payload(decode=False)
                    break
        else:
            body = email.get_payload(decode=False)

        
        corpus.append(body.decode("ISO-8859-1"))
        y.append(1)


#print len(corpus)
#print y
#print corpus


vectorizer = CountVectorizer(min_df=1,max_features=10000)

# fit_transform(): Learn the vocabulary dictionary and return term-document matrix.
X = vectorizer.fit_transform(corpus)

#print X

X_array = X.toarray()

#print X_array

bool_array = X_array.astype(bool)
#print bool_array

binary_array = bool_array.astype(int)
#print binary_array

#email_one = binary_array[0]
#print email_one

# get_feature_names(): Array mapping from feature integer indices to feature name
features = vectorizer.get_feature_names()

#print features


# In[85]:

my_y = pd.DataFrame(y, dtype = 'uint8')
my_y.to_csv('Labels.csv', index=False, header=False)

my_feat = pd.DataFrame(features, dtype = 'str')
my_feat.to_csv('Feature_Names.csv', index=False, header=False)

my_df = pd.DataFrame(X.toarray(), dtype = 'uint8')
my_df.to_csv('Features.csv', index=False, header=False)

my_binary_df = pd.DataFrame(binary_array, dtype = 'uint8')
my_binary_df.to_csv('BinaryFeatures.csv', index=False, header=False)

#print my_y
#print my_feat
#print my_df
#print my_binary_df

# Check the dimensions
#print X_array.shape
#print len(y)



