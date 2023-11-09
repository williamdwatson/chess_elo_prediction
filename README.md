# Chess Elo prediction
This repository contains a PyTorch transfomer-based approach for predicting chess Elo based on only the sequence of algebraic moves. The default training data are from [Lichess](https://database.lichess.org/).

## Training
1. Clone the repo and navigate to it
2. Obtain a list of zstandard-compressed PGN files to train on and save under the name `game_list.txt`. The default is a list of all games from January 2013 through September 2023 from Lichess.
3. Change the settings in `settings.py`.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. `FILE_DIR` - where the data is stored. This location should have enough storage for all the decompressed files.  
    &nbsp;&nbsp;&nbsp;&nbsp;b. `EMBEDDING_DIM` - the number of dimensions used for the move embedding (default=72)  
    &nbsp;&nbsp;&nbsp;&nbsp;c. `SEQUENCE_LENGTH` - the maximum number of moves permitted in a game (default=150)  
4. Download and process the data using `download_data.py`. The parameters passed to `parse_file` can be modified to filter the data.  
    a. By default the data is downloaded and parsed using five processes, before being combined and split into training, validation, and testing sets.
5. Run either `create_indices.py` to generate an index mapping for every unique move (this will cause training to use a trainable `Embedding` layer), or run `create_embeddings.py` to use `Word2Vec` to pretrain embeddings.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. Pretrained embeddings from `Word2Vec` are `embeddings.model` and `move_vecs.wordvectors`
7. Run the training using either `train.py` or `train_ddp.py`. This defaults to using trainable embeddings, and the setup may require tuning for the available memory. The current settings are for a setup with 4 RTX 2080 Tis with 11 GB of memory each.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. `train.py` will use PyTorch's `DataParallel` to run on multiple GPUs if applicable, otherwise it will target the GPU or CPU depending on availability.  
    &nbsp;&nbsp;&nbsp;&nbsp;b. `train_ddp.py` uses PyTorch's `DistributedDataParallel` to run on multiple GPUs. However, this was slower than the simpler `DataParallel` approach.

## Inference
*Work in progress*

## Performance
*Work in progress*

## How it works
*Work in progress*
