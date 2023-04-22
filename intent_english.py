import spacy
from spacy.lang.en import English
from spacy import displacy
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_sm")

train_data = [("I want to order a pizza", {"intent": "ORDER_PIZZA"}),
              ("Can you recommend a good restaurant?", {"intent": "RECOMMEND_RESTAURANT"}),
              ("What time does the train leave?", {"intent": "TRAIN_SCHEDULE"}),
              ("How do I get to the airport?", {"intent": "DIRECTIONS_TO_AIRPORT"}),
              ("What's the weather like today?", {"intent": "WEATHER_FORECAST"})]

# Define the training loop
def train(model, train_data, epochs):
    for epoch in range(epochs):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            model.update(texts, annotations)

# Train the model
train(nlp, train_data, 10)

# Test the model
doc = nlp("I would like to order a pizza")
print(doc.cats)
