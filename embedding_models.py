from sentence_transformers import SentenceTransformer

SUPPORTED_MODELS = ['multi-qa-MiniLM-L6-cos-v1', 'multi-qa-distilbert-dot-v1', 'multi-qa-mpnet-base-dot-v1']

class SentenceTransformerEmbeddingModel:
    def __init__(
        self,
        model_name: str,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise Exception('unsupported embedding model')
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return self.model.encode(text)
