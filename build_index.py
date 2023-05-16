from embedding_models import EmbeddingModel
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import argparse
import faiss

def get_args():
    parser = argparse.ArgumentParser(description="Build index")
    parser.add_argument(
        "-d",
        "--data-dir",
        help="path to folder containing text",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="out",
        help="file to output index mapping to"
    )
    parser.add_argument(
        "--model_source",
        default="sentencetransformer",
        help="type of embedding model to use"
    )
    parser.add_argument(
        "--model_name",
        default="multi-qa-MiniLM-L6-cos-v1",
        help="specific pretrained model to use"
    )
    parser.add_argument(
        "--tokenizer_path",
        default="none",
        help="tokenizer path if applicable"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        help="batch size"
    )
    return parser.parse_args()

def build_index(data_folder, outfile, model_source, model_name, tokenizer_path, batch_size):

    chunks = []
    # TODO: come up with actual chunking strategy
    for file in tqdm(os.listdir(data_folder)):
        try:
            text = open(data_folder + '/' + file).read()
        except:
            print('error processing', file, 'skipping')
            continue
        buffer = ""
        for i in text.split('\n'):
            buffer += i + '\n'
            if len(buffer) >= 1024:
                chunks.append(buffer.strip())
                buffer = ""
        if buffer:
            chunks.append(buffer.strip())

    model = EmbeddingModel(model_source, model_name, tokenizer_path)
    embeddings = model.encode(chunks, batch_size=batch_size)

    if model_source == 'sentencetransformer':
        embedding_dim = model.model.get_sentence_embedding_dimension()
    else:
        embedding_dim = len(embeddings[0])

    # TODO: add more faiss options
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    print(index.ntotal, 'document shards indexed.')

    faiss.write_index(index, outfile + '.bin')
    pd.DataFrame(chunks, columns=['text']).to_csv(outfile + '.csv', index=False)

if __name__ == "__main__":
    args = get_args()
    build_index(args.data_dir,
                args.outfile,
                args.model_source,
                args.model_name,
                args.tokenizer_path,
                int(args.batch_size))