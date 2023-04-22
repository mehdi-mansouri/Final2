from stanfordnlp import StanfordNLP

# Initialisieren des StanfordNLP-Pipelines
nlp = StanfordNLP()

# Beispieltext
text = "Barack Obama was born in Hawaii and was the 44th president of the United States."

# Verarbeitung des Texts mit StanfordNLP
doc = nlp(text)

# Extrahieren der benannten Entitäten
entities = [(sent.entities[i].text, sent.entities[i].type) for sent in doc.sentences for i in range(len(sent.entities))]

# Ausgabe der Entitäten
for entity in entities:
    print(entity)
