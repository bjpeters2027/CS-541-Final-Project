from src.utils.enums.embedding_enums import Embedding

word2vec = None  # Placeholder for the word to vector mapping for the Word2Vec model
vec2word = None  # Placeholder for the vector to word mapping for the Word2Vec model
word2bert = None  # Placeholder for the word to vector mapping for the BERT model
bert2word = None  # Placeholder for the vector to word mapping for the BERT model

def word2vector(model, word):
    """
    Function that converts word to vector using the provided model.
    """
    if model == Embedding.WORD2VEC:
        return word2vec[word]
    elif model == Embedding.BERT:
        return word2bert[word]
    
def vector2word(model, vector):
    """
    Function that converts vector to word using the provided model.
    """
    if model == Embedding.WORD2VEC:
        return vec2word[vector]
    elif model == Embedding.BERT:
        return bert2word[vector]