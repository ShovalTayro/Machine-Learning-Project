import string
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()
stoplist = stopwords.words('english')

def clean(s):
    # Remove leading and trailing whitespace
    s = s.strip()
    # Remove unwanted words
    out = ' '.join(word for word in s.split() if not word in stoplist)
    # Stemming
    sentence = ' '.join(ps.stem(word) for word in out.split())
    # Remove punctuation in the string
    return ''.join(ch for ch in sentence if ch not in string.punctuation)


df = pd.read_csv('Suicide_Detection_new.csv', header=0, encoding='unicode_escape')
index = 0
text = df['text'].tolist()
for i in range(len(text)):
    print(index)
    line = df.loc[i, 'text']
    line = re.sub("([^A-Za-z0-9 ])+", "", line)
    df.loc[i, 'text'] = clean(line.lower())
    index = index + 1
df = df.dropna()
df.to_csv(r'clean_dataset.csv')
