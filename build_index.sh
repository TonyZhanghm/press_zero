python build_index.py \
    -d /persist/sguo/hive_docs \
    -o test \
    --model_source local \
    --model_name /persist/sguo/colbert/colbert-embeddings/1/model.pt \
    --tokenizer_path /persist/sguo/colbertv2.0 \
    --batch_size 32