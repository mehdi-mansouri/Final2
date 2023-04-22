from cgitb import text
import os
import gensim
import pandas as pd
class Doc2vecClass:
    global model 
    def __init__(self, userTextVariable,existdoc):
        self.userTextVariable = userTextVariable
        self.existdoc=existdoc
        #self.model=gensim.models
        #self.train_corpus

    def read_corpus(self,fname, tokens_only=False):
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

    def existDoc2vec(self,sentiment):
          #insert_to_db("EnergyTips1.json")

        model = gensim.models.doc2vec.Doc2Vec(vector_size=150, min_count=1, epochs=200 ,workers=8)
        
        if (sentiment == 'orginal'):
            train_corpus = list(self.read_corpus("EnergyTipss.json"))
            fname = "my_doc2vec_model_orginal"
        elif (sentiment == 'posetive'):
            train_corpus = list(self.read_corpus("EnergyTipss_posetive_tweets.json"))
            fname = "my_doc2vec_model_posetive"
        elif (sentiment == 'negative'):
            train_corpus = list(self.read_corpus("EnergyTipss_negative_tweets.json"))
            fname = "my_doc2vec_model_negative"
        elif (sentiment == 'neutral'):
            train_corpus = list(self.read_corpus("EnergyTipss_neutral_tweets.json"))
            fname = "my_doc2vec_model_neutral"
        tweet_liste=[]
        #wenn einmal model trainiert wurde, brauchen wir nicht nochmal machen,
        #es ist notwendig neue model erstellen ,wenn wir nue datei erstellt wird 
        if self.existdoc:
            #test_corpus = list(self.read_corpus("save_file.json", tokens_only=True))
            
            model.build_vocab(train_corpus)
            #print(f"Word 'penalty' appeared {model.dv.get_vecattr('energy', 'count')} times in the training corpus.")
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs )
            #model.test(test_corpus, total_examples=model.corpus_count, epochs=model.epochs )
            #----save model
           
            model.save(fname)
            
        else: 
            model = gensim.models.doc2vec.Doc2Vec.load(fname)
        inferred_vector = model.infer_vector(self.userTextVariable)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('1-MOST', 0), ('2-MOST', 1),('3-MOST', 2),('4-MOST', 3),('5-MOST', 4),
                            ('6-MOST', 5),('7-MOST', 6),('8-MOST', 7),('9-MOST', 8),('10-MOST', 9), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
            tweet_liste.append(' '.join(train_corpus[sims[index][0]].words))
        return(tweet_liste)
    
    def userText(self,sentiment):
        return self.existDoc2vec(sentiment)

