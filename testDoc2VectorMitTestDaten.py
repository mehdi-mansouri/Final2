from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize



# Liste von Dokumenten als Text
documents = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third document.",
    "Is this the first document?"
]

# Konvertieren von Dokumenten in TaggedDocument-Objekte
tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

# Aufteilen von tagged_docs in Test- und Trainingsdaten
test_docs = tagged_docs[:2]
train_docs = tagged_docs[2:]

# Trainieren des Modells auf den Trainingsdaten
model = doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_docs)
model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)
print(model)
# Bewerten der Genauigkeit des Modells auf den Testdaten
inferred_vectors = [model.infer_vector(word_tokenize(doc.words)) for doc in test_docs]
print(inferred_vectors)