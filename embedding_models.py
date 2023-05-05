from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import tiktoken
import openai

SUPPORTED_MODELS = {
    'sentencetransformer': ['multi-qa-MiniLM-L6-cos-v1', 'multi-qa-distilbert-dot-v1', 'multi-qa-mpnet-base-dot-v1'],
    'huggingface': ['bert-base-uncased', 'xlm-roberta-base', 'roberta-base', 't5-base'],
    'openai': ['text-embedding-ada-002']
    }

class EmbeddingModel:
    """
    Wrapper class for embedding models.
    """
    def __init__(
        self,
        model_source: str,
        model_name: str,
        aggregate_strategy: str = 'mean'
    ):
        """
        Args:
            model_source (str):
                Source of embedding model. Choose from ['sentencetransformer', 'huggingface', 'openai'].
            model_name (str):
                Name of specific model to use.
            aggregate_strategy (str):
                Only applies to BERT-style huggingface models. Use 'mean' to average all token embeddings and 'cls' to use the class token.

        """
        if model_source not in ['sentencetransformer', 'huggingface', 'openai']:
            raise Exception("unrecognized model source '" + model_source + "'")
        if model_name not in SUPPORTED_MODELS[model_source]:
            raise Exception('unsupported ' + model_source + ' embedding model')

        self.model_source = model_source
        self.aggregate_strategy = aggregate_strategy

        if model_source == 'sentencetransformer':
            self.model = SentenceTransformer(model_name, device='cuda')
        elif model_source == 'huggingface':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if 't5' in model_name:
                # Extract encoder from T5 architecture
                self.model = self.model.encoder
            self.model = self.model.cuda()

    # TODO: support batch encoding
    def encode(self, text):
        if self.model_source == 'sentencetransformer':
            return self.model.encode(text)
        elif self.model_source == 'huggingface':
            tokens = self.tokenizer(text, return_tensors='pt').to('cuda')
            output = self.model(**tokens)
            if self.aggregate_strategy == 'mean':
                return torch.mean(output['last_hidden_state'], dim=1).cpu().detach().flatten().numpy()
            elif self.aggregate_strategy == 'cls':
                return output['pooler_output'].cpu().detach().flatten().numpy()
            else:
                raise Exception ('unsupported aggregate strategy')
        elif self.model_source == 'openai':
            raise Exception ('not implemented yet')
            # TODO: add openai API support