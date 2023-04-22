#from curses import window
from cgitb import text
import os
import gensim
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
# from nltk.stem  import WordNetLemmatizer
# from nltk.corpus import stopwords
# import seaborn as sns
import mysql.connector
# import numpy as np
import pandas as pd
# import logging

def read_corpus(fname, tokens_only=False):
    df = pd.read_json(fname, encoding="utf-8")
    corpus=df['tweet'].tolist()
    #print(corpus[0])
    corpus=[doc.lower() for doc in corpus]
    for i,doc in enumerate(corpus):
        doc=doc.encode("utf-8")
        tokens = gensim.utils.simple_preprocess(doc)
        #print(doc)
        #print(tokens)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens,[i])


#insert_to_db("EnergyTips1.json")
train_corpus = list(read_corpus("EnergyTipss.json"))
                # test_corpus = list(read_corpus("save_file.json", tokens_only=True))

                # model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=200 ,workers=8)
                # model.build_vocab(train_corpus)
                # #print(f"Word 'penalty' appeared {model.dv.get_vecattr('energy', 'count')} times in the training corpus.")
                # model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs )
                # #model.test(test_corpus, total_examples=model.corpus_count, epochs=model.epochs )


                # #----save model

fname = "my_doc2vec_model"
                # model.save(fname)
model = gensim.models.doc2vec.Doc2Vec.load(fname)


vector = model.infer_vector(['how', 'tips', 'save' 'energy'])
#dfvector=pd.DataFrame(model.dv.vectors,columns=['x1','x2'])
#print(dfvector)
#sns.scatterplot(data =dfvector,x='x1',y='x2')
#print(dfvector.loc[dfvector.x1 >13])



# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)
#     second_ranks.append(sims[1])

# import collections

# counter = collections.Counter(ranks)
#print(counter)
inferred_vector = model.infer_vector(['energy','save' ,'bathroom'])
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
doc_id=555

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('1-MOST', 0), ('2-MOST', 1),('3-MOST', 2),('4-MOST', 3),('5-MOST', 4),
                     ('6-MOST', 5),('7-MOST', 6),('8-MOST', 7),('9-MOST', 8),('10-MOST', 9), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

print('-------------------------')

# Pick a random document from the corpus and infer a vector from the model
#------------------------
# import random
# doc_id = random.randint(0, len(train_corpus) - 1)

# # Compare and print the second-most-similar document
# print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# sim_id = second_ranks[doc_id]
# print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))
#----------------------

print('-------------------------')
# Pick a random document from the test corpus and infer a vector from the model


#------------------------------
# doc_id = random.randint(0, len(test_corpus) - 1)
# inferred_vector = model.infer_vector(test_corpus[doc_id])
# sims = model.dv.most_similar([vector], topn=len(model.dv))
#-----------------------------
# Compare and print the most/median/least similar documents from the train corpus

#print(vector)

# print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))#


