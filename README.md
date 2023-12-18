# Chess Elo prediction
This repository contains a PyTorch transfomer-based approach for predicting chess Elo based on only the sequence of algebraic moves. The default training data are from [Lichess](https://database.lichess.org/).

## Training
1. Clone the repo and navigate to it
2. Obtain a list of zstandard-compressed PGN files to train on and save under the name `game_list.txt`. The default is a list of all games from January 2013 through September 2023 from Lichess.
3. Change the settings in `settings.py`.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. `FILE_DIR` - where the data is stored. This location should have enough storage for all the decompressed files (at least several terabytes).  
    &nbsp;&nbsp;&nbsp;&nbsp;b. `EMBEDDING_DIM` - the number of dimensions used for the move embedding (default=72)  
    &nbsp;&nbsp;&nbsp;&nbsp;c. `SEQUENCE_LENGTH` - the maximum number of moves permitted in a game (default=150)  
4. Download and process the data using `download_data.py`. The parameters passed to `parse_file` can be modified to filter the data.  
    a. By default the data is downloaded and parsed using five processes, before being combined and split into training, validation, and testing sets.
5. Run either `create_indices.py` to generate an index mapping for every unique move (this will cause training to use a trainable `Embedding` layer), or run `create_embeddings.py` to use `Word2Vec` to pretrain embeddings.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. Pretrained embeddings from `Word2Vec` are `embeddings.model` and `move_vecs.wordvectors`; note that these embeddings are of length 100.
7. Run the training using either `train.py` or `train_ddp.py`. This defaults to using trainable embeddings, and the setup may require tuning for the available memory. The current settings are for a setup with 4 RTX 2080 Tis with 11 GB of memory each.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. `train.py` will use PyTorch's `DataParallel` to run on multiple GPUs if applicable, otherwise it will target the GPU or CPU depending on availability.  
    &nbsp;&nbsp;&nbsp;&nbsp;b. `train_ddp.py` uses PyTorch's `DistributedDataParallel` to run on multiple GPUs. However, this was slower than the simpler `DataParallel` approach.  
    &nbsp;&nbsp;&nbsp;&nbsp;c. The default model is `modelAvg` - see the [How it works](#how-it-works) section for more details
8. Training progress can be checked using `check_training.py` which will generate basic plots to check the current training status
9. After training, run `generate_val_results.py` and then `get_percentiles.py` to run validation scripts.
10. If desired, run `convert_to_onnx.py` to create ONNX version of the models.

## Inference
*Work in progress*

## Performance
*Work in progress*

## How it works
### The data
#### Downloading and parsing the data
1. The raw data should be as [PGN](http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm) (Portable Game Notation) files. These are looped through line by line, and the game year/month, time (both base and bonus), black and white Elos, and move sequences parsed out. The result of the game and whether it ended in a checkmate are also parsed out, although ideally the model shouldn't need this information.  
    &nbsp;&nbsp;&nbsp;&nbsp;a. The data will be parsed with 5 processes (more can cause the Lichess server to refuse to respond). Each process will write to a different text file; every game will be on a separate line. The downloaded and decompressed files both use temporary files.  
    &nbsp;&nbsp;&nbsp;&nbsp;b. After all the data is downloaded, the first file will be split in two for the validation and testing data, and the other 4 will be combined into training. The original files will be deleted afterwards.  
#### Exploring the data
2. How stationary the Elo target is can be checked in `check_year_scores.py` - this loops through two years (default 2015 and 2022) and grabs all games with identical move sequences and compares the Elos.  
3. `create_indices.py` generates a JSON mapping of every unique move to an integer index; `create_onehot.py` does the same thing except each value is now a list with zeros everywhere except a 1 at the index.  
4. `create_embeddings.py` uses [`gensim's Word2Vec`](https://radimrehurek.com/gensim/models/word2vec.html) to create pretrained embeddings of each move using a window size of 3 and an embeddings size of 72. This isn't the approach used in the actual models, so is just there as legacy code. 
### Models
There are three model options in `train.py` (`train_ddp.py` is an older version and has not been updated with all options):
#### modelFlatten
This model first passes the sequence of moves through an Embedding layer and then a Transformer Encoder. The results are then flattened and passed through linear layers until the black and white Elo are predicted.

This approach is far slower and larger than it needs to be.
#### modelAvg
This model first passes the sequence of moves through an Embedding layer and then a TransformerEncoder. It then takes an mean down the embedding dimension (so the result is the same length as the sequence) and passes the result through linear layers until the black and white Elo are predicted.
#### modelTime
This model is very similar to [modelAvg](#modelavg), except that after the mean it concatenates the base and bonus time values prior to the linear layers.
#### modelLinear
This model first passes the sequence of moves through an Embedding layer. It then flattens the result and passes that through linear layers until the black and white Elo are predicted.

This approach is faster to train and achieves similar results to the transformer-based approachs, but results in orders-of-magnitude more weights.
