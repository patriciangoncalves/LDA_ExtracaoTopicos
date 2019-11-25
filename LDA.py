"""
Data criação: 01/09/2019
@author: patricia_ngoncalves@sicredi.com.br

Aplicando LDA - Latent Dirichlet Allocation - para Extração de Tópicos em Notícias

"""

import nltk
import gensim
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from unidecode import unidecode

documents = open('Sustentabilidade.txt', encoding="utf-8").read()
S=nltk.corpus.stopwords.words('portuguese')
sentencas = sent_tokenize(documents, language='portuguese')
print(sentencas)

def preprocess(text):
    result = []
    p_stemmer = PorterStemmer()
    for token in gensim.utils.simple_preprocess(text):
        if token not in S and len(token) > 3:
            token=unidecode(token)
            token=p_stemmer.stem(token)
            result.append(token)
    return result

processed_docs = preprocess(documents)
print(processed_docs)
dictionary = gensim.corpora.Dictionary([processed_docs])

bow_corpus = [dictionary.doc2bow(doc) for doc in [processed_docs]]

lda_model = gensim.models.LdaModel(bow_corpus, num_topics=5, id2word=dictionary, passes=10)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
