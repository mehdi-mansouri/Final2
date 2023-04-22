import spacy


class TextArbeiten():
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize_text(self, text):
        # Tokenisierung des Textes
        doc = self.nlp(text)

        # Erstellung einer Liste von Tokens
        tokens = [token.text for token in doc]

        return tokens

    def lemmatize_text(self, tokens):
        # Lemmatisierung der Tokens
        lemmas = [token.lemma_ for token in self.nlp(' '.join(tokens))]
        print(lemmas)
        return lemmas

    def remove_stopwords(self, tokens):
        # Entfernen der Stopwörter
        tokens_without_stopwords = [
            token for token in tokens if not self.nlp.vocab[token].is_stop]
        text_without_stopwords = ' '.join(tokens_without_stopwords)

        return text_without_stopwords

    def text_bearbeiten(self):
        tokens = self.tokenize_text(self.text)
        lemmas = self.lemmatize_text(tokens)
        text_without_stopwords = self.remove_stopwords(lemmas)

        return text_without_stopwords
