from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
import torch
import tiktoken
import openai
import os

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
        tokenizer_path: str='none',
        aggregate_strategy: str = 'mean'
    ):
        """
        Args:
            model_source (str):
                Source of embedding model. Choose from ['sentencetransformer', 'huggingface', 'openai', 'local'].
            model_name (str):
                Name of specific model to use.
            tokenizer_path (str):
                Separate path to tokenizer (only affects local models).
            aggregate_strategy (str):
                Only applies to BERT-style huggingface models. Use 'mean' to average all token embeddings and 'cls' to use the class token.

        """
        if model_source not in ['sentencetransformer', 'huggingface', 'openai', 'local']:
            raise Exception("unrecognized model source '" + model_source + "'")
        if model_source != 'local' and model_name not in SUPPORTED_MODELS[model_source]:
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
        elif model_source == 'local':
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = torch.jit.load(model_name)

    # TODO: support batch encoding for huggingface
    # TODO: add DataParallel
    def encode(self, text: Union[str, List[str]], batch_size: int):

        if self.model_source == 'sentencetransformer':
            return self.model.encode(text, batch_size=batch_size)
        
        elif self.model_source == 'local':
            embeddings = []
            for i in range(0, len(text), batch_size):
                batch = text[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt', padding="max_length", truncation=True).to('cuda')
                embeddings.append(self.model(inputs['input_ids'], inputs['attention_mask'])) #.hidden_states[-1][:, :1].squeeze(1)
            return torch.cat(embeddings).cpu().numpy()
        
        elif self.model_source == 'huggingface':
            tokens = self.tokenizer(text, return_tensors='pt', padding=True).to('cuda')
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
