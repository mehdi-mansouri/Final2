import gensim
import pandas as pd
class Doc2vecClass:
    global model 
    def __init__(self, userTextVariable,existdoc):
        self.userTextVariable = userTextVariable
        self.existdoc=existdoc

    def read_corpus(self,fname):
        df = pd.read_json(fname, encoding="utf-8")
        corpus=df['contenttext'].tolist()
        corpus=[doc.lower() for doc in corpus]
        for i,doc in enumerate(corpus):
            doc=doc.encode("utf-8")
            tokenize=tokenize_text(doc)
            lemma=lemmatize_text(tokenize)
            text=remove_stopwords(lemma)
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(text,[i])

    def existDoc2vec(self,Sentiment):

        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=200 ,workers=8)
        
        if(Sentiment =='Posetive'):
            train_corpus = list(self.read_corpus("EnergyTipss_posetive_tweets.json"))
            orginal_file="EnergyTipss_posetive_tweets.json"
            model_fname = "Posetive_doc2vec_model"
        elif(Sentiment =='Negative'):
            train_corpus = list(self.read_corpus("EnergyTipss_negative_tweets.json"))
            orginal_file="EnergyTipss_negative_tweets.json"
            model_fname = "Negative_doc2vec_model"
        elif(Sentiment =='Neutral'):
            train_corpus = list(self.read_corpus("EnergyTipss_neutral_tweets.json"))
            orginal_file="EnergyTipss_neutral_tweets.json"
            model_fname = "Neutral_doc2vec_model"
        else:
            train_corpus = list(self.read_corpus("EnergyTips.json"))
            orginal_file="EnergyTips.json"
            model_fname = "all_doc2vec_model"

        tweet_liste=[]

        if self.existdoc:
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs )
            model.save(model_fname)  
        else: 
            model = gensim.models.doc2vec.Doc2Vec.load(model_fname)

        tokenize=tokenize_text(self.userTextVariable)
        lemma=lemmatize_text(tokenize)
        userText=remove_stopwords(lemma)
        inferred_vector = model.infer_vector(userText)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('1-MOST', 0), ('2-MOST', 1),('3-MOST', 2),('4-MOST', 3),('5-MOST', 4),
                            ('6-MOST', 5),('7-MOST', 6),('8-MOST', 7),('9-MOST', 8),('10-MOST', 9), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
            tweet_liste.append(' '.join(train_corpus[sims[index][0]].words))
        return(tweet_liste)
    
    def userText(self,setiment):
        return self.existDoc2vec(setiment)



