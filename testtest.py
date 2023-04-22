

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Extraktion der Entitäten
for ent in doc.ents:
    print(ent.text, ent.label_)
