# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np

import glob
from email.parser import Parser
parser = Parser()

from io import open

corpus = []
y = []

for filepath in glob.glob('../Datasets/enron/ham/*'): 
    with open(filepath, 'r', encoding ='ISO-8859-1') as ofile:
        email = parser.parse(ofile)
        
        ## code from:
        ## https://stackoverflow.com/questions/17874360/python-how-to-parse-the-body-from-a-raw-email-given-that-raw-email-does-not/32840516#32840516
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

        
        corpus.append(body)
        y.append(0)

for filepath in glob.glob('../Datasets/enron/spam/*'): 
    with open(filepath, 'r', encoding ='ISO-8859-1') as ofile:
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

        
        corpus.append(body)
        y.append(1)

#vectorizer = CountVectorizer(min_df=1, stop_words='english')
vectorizer = CountVectorizer(min_df=1, stop_words='english', max_features=1000)
X = vectorizer.fit_transform(corpus)

my_y = pd.DataFrame(y, dtype = np.uint8)
my_y.to_csv('../Datasets/EmailDataProcessed/Labels.csv', index=False, header=False)

my_feat = pd.DataFrame(vectorizer.get_feature_names())
my_feat.to_csv('../Datasets/EmailDataProcessed/Feature_Names.csv', encoding='utf-8', index=False, header=False)

my_df = pd.DataFrame(X.toarray(), dtype = np.uint8)
my_df.astype(bool).astype(np.uint8).to_csv('../Datasets/EmailDataProcessed/Features.csv', header=False, index=False)

#print my_y
#print my_feat
#print my_df

no_emails, no_features = X.toarray().shape

# Check the dimensions
print 'Number of emails:', no_emails
print 'Number of features:', no_features
#print len(y)



