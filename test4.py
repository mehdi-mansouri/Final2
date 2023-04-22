import stanza

# Laden des englischen Modells
nlp = stanza.Pipeline('en', processors='tokenize,ner')

# Beispieltext
text = "Apple is looking at buying U.K. startup for $1 billion"

# Verarbeitung des Textes durch Stanza
doc = nlp(text)

# Extrahieren von Entitäten
for ent in doc.entities:
    print(f"{ent.text}\t{ent.type}")
