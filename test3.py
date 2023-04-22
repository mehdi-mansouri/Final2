import allennlp_models.tagging

from allennlp.predictors import Predictor
from allennlp_models.tagging import FineGrainedNERPredictor

# download the fine-grained NER model
model_url = "https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-06-15.tar.gz"
predictor = Predictor.from_path(model_url)

# predict entities in a sentence
result = predictor.predict(
    sentence="John Smith lives in the United States and works for Microsoft Research."
)

print(result['tags'])
