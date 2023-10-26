# Chess Elo prediction
This repository contains a PyTorch transfomer-based approach for predicting chess Elo based on only the sequence of algebraic moves. The default training data are from [Lichess](https://database.lichess.org/).

## Training
1. Clone the repo and navigate to it
2. Obtain a list of zstandard-compressed PGN files to train on and save under the name `game_list.txt`. The default is a list of all games from January 2013 through September 2023 from Lichess.
3. Change the settings in `settings.py`.  
    a. `FILE_DIR` - where the data is stored. This location should have enough storage for all the decompressed files.
    b. `EMBEDDING_DIM` - the number of dimensions used for the move embedding (default=72)
    c. `SEQUENCE_LENGTH` - the maximum number of moves permitted in a game (default=150)
4. Download and process the data using `download_data.py`. The parameters passed to `parse_file` can be modified to filter the data.  
    a. By default the data is downloaded and parsed using five processes, before being combined and split into training, validation, and testing sets.
5. Run either `create_indices.py` to generate an index mapping for every unique move (this will cause training to use a trainable `Embedding` layer), or run `create_embeddings.py` to use `Word2Vec` to pretrain embeddings.
6. Run the training using `train.py`. This defaults to using trainable embeddings, and the setup may require tuning for the available memory. The current settings are for a setup with 4 10-GB GPUs.

## Inference
*Work in progress*

## Performance
*Work in progress*

## How it works
*Work in progress*