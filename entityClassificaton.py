import spacy
import pandas as pd

class entityClass:
    def __init__(self, tweet):
        self.tweet = tweet

    def entityspacy(self):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.tweet)
        entities = []
        for entity in doc.ents:
            entities.append({"text":entity.text,"label":entity.label_})
        return entities
 
