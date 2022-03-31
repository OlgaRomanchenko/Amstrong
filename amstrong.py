import csv
from string import punctuation
from requests_html import HTMLSession
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams
from nltk.stem.snowball import EnglishStemmer

url = 'https://en.wikipedia.org/wiki/Neil_Armstrong'

session = HTMLSession()
response = session.get(url)
text_body = response.html.xpath('//p')

result = ''
for i in text_body:
    result += i.text

stemmer = EnglishStemmer()
STOP_WORDS = set(stopwords.words('english'))
words = word_tokenize(result)


cleaned_words = []
for word in words:
    if (word.lower() in STOP_WORDS) or (word.lower() in punctuation):
        continue
    cleaned_words.append(word.lower())

tag_get = pos_tag(words)

nnp_words = {}

for word, tag in tag_get:
    st_word = stemmer.stem(word)
    if (st_word in STOP_WORDS) or (st_word in punctuation):
        continue
    if tag == 'NNP':
        if word in nnp_words:
            nnp_words[word] += 1
        else:
            nnp_words[word] = 1
    else:
        continue

nnp_frequency = sorted(nnp_words.items(), key=lambda x: x[1], reverse=True)
nnp_frequency = dict(nnp_frequency)

with open('NNPwors.csv', 'w', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    [f.write(f'{key},{value} \n') for key, value in nnp_frequency.items()]
    print('frequency')


phrases = list(bigrams(cleaned_words))
tag_gets =list(bigrams(pos_tag(cleaned_words)))

b_words = []

for word,tag in tag_gets:
    st_word = stemmer.stem(word[0])
    if (st_word in STOP_WORDS) or (st_word in punctuation):
        continue
    if word[1] == 'NNP':
        b_words.append((word[0], tag[0]))
    else:
        continue


b_words = [' '.join(gr) for gr in b_words]
b_words_sorted = sorted(b_words, key=lambda x: x[0])
with open('sorted_NNP.txt', 'a', encoding='utf-8') as f:
    for i in b_words_sorted:
        f.write(f'{i} \n')
    print('Work completed.\
    Formatted bigrams are written in txt. file')
