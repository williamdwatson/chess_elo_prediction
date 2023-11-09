import json, os, pickle
import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from itertools import islice
from torch.utils.data import DataLoader, get_worker_info, IterableDataset
from gensim.models import KeyedVectors
from numbers import Number
from typing import Dict, Generator, Iterator, List, Literal, Optional, Tuple
from tqdm import tqdm
from settings import FILE_DIR

#: Number of embedding dimensions for each move
EMBEDDING_DIM = 72
#: Maximum number of moves in a game
SEQUENCE_LENGTH = 150
#: Number of unique moves
NUMBER_OF_MOVES = 9839

def check_move_only(mv: str) -> bool:
    """
    Checks if a given string contains only characters allowed in a move

    Parameters
    ----------
    mv : str
        String to check
    
    Returns
    -------
    bool :
        Whether `mv` contains any banned characters
    """
    return not any(bad in mv for bad in ('.', '[', ']', '{', '}', '%', ':'))

def prune_to_move(mv: str) -> str:
    """
    Prunes extraneous characters from a move

    Parameters
    ----------
    mv : str
        String to check
    
    Returns
    -------
    str :
        `mv` with various extraneous characters removed
    """
    return mv.replace('!', '').replace('?', '').replace('#', '').replace(',', '').replace('\n', '').strip()

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    print('Using CPU only training - this may be very slow')
    torch.device('cpu')

class ChessDataset(IterableDataset):
    
    def __init__(self, fname: str, num_workers: int, worker_id: int, total: int, move_vec_file: Optional[str]=None, one_hot_file: Optional[str]=None, indices_file: Optional[str]=None, use: Literal['pretrained', 'onehot', 'indices']='indices') -> None:
        """
        Pytorch iterable Dataset for the chess data
        
        Parameters
        ----------
        fname : str
            Path to the file containing the game data
        num_workers : int
            The number of workers
        worker_id : int
            The id of the Dataset's worker
        total : int
            Total length of the game data
        move_vec_file : str, optional, default=None
            File containing move embeddings; must be specified if `use` is "pretrained"
        one_hot_file : str, optional, default=None
            File containing one-hot encodings; must be specified if `use` is "onehot"
        indices_file : str, optional, default=None
            File containing move indices; must be specified if `use` is "indices"
        use : str, default='indices'
            Whether to return an embedding from a pretrained set ('pretrained') or return the move index ('onehot' or 'indices')
        """
        super().__init__()
        self.fname = fname
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.total = total
        self.use_pretrained_embedding = use == 'pretrained'
        if use == 'pretrained':
            self.wv = KeyedVectors.load(move_vec_file)
        elif use == 'onehot':
            with open(one_hot_file, 'r') as f:
                # Get the index of the `1` in each one-hot encoding (and add 1 since 0 will be the padding index for the `Embedding` layer)
                self.indices: Dict[str, int] = {mv: onehot.index(1)+1 for mv, onehot in json.load(f).items()}
        else:
            with open(indices_file, 'r') as f:
                self.indices: Dict[str, int] = json.load(f)
    
    def get_embedding(self, move_list: List[str]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Converts a string of moves into an embedding
        
        Parameters
        ----------
        move_list : list
            List of string chess moves
        
        Returns
        -------
        arr : np.ndarray
            Numpy array of the embedding representation of `move_list`
        mask : np.ndarray
            Numpy array of the mask of `arr` corresponding to the first dimension of `arr`;
            this is True where embeddings are present, False where they are not (i.e. beyond the length of `move_list`)
        """
        arr = np.zeros((SEQUENCE_LENGTH, EMBEDDING_DIM), dtype=np.float32)
        arr[:len(move_list)] = [self.wv[move] for move in move_list]
        mask = np.ones(SEQUENCE_LENGTH, dtype=np.bool_)
        mask[:len(move_list)] = False
        return arr, mask

    def get_indices(self, move_list: List[str]) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """
        Converts a string of moves into an array of indices

        Parameters
        ----------
        move_list : list
            List of string chess moves
        
        Returns
        -------
        arr : np.ndarray
            Numpy array of the indices of each move (with 0 being padding)
        mask : np.ndarray
            Numpy array of the mask of `arr`;
            this is True where indices are present, False where they are not (i.e. beyond the length of `move_list`)
        """
        arr = np.zeros(SEQUENCE_LENGTH, dtype=np.int64)
        arr[:len(move_list)] = [self.indices[mv] for mv in move_list]
        mask = np.ones(SEQUENCE_LENGTH, dtype=np.bool_)
        mask[:len(move_list)] = False
        return arr, mask

    def __iter__(self) -> Generator[Tuple[int, int, float, float, int, int, Tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]], None, None]:
        """
        Yields data from the dataset, including support for multiple workers if requested (although `total` must have been specified upon initialization).
        Note that using multiple workers uses `islice`, so the startup time may be long with multiple workers.

        Yields
        ------
        black_elo : int
            Black Elo
        white_elo : int
            White Elo
        time_base : float
            Base time of the game in seconds
        time_bonus : float
            Bonus time of the game in seconds
        result : int
            1 if white won, 2 if black one, 4 if a tie
        checkmate : int
            1 if checkmate was reached, 0 if the game ended another way
        tuple :
            Length-2 tuple of the indices of the moves as a SEQUENCE_LENGTH-long int64 numpy array, and the embedding length mask as a SEQUENCE_LENGTH-length boolean numpy array
        """
        if self.total is None:
            raise ValueError('`total` must be specified upon initialization to use multiple workers')
        split_indices = np.linspace(0, self.total, self.num_workers+1, dtype=int, endpoint=True)
        start_idx = split_indices[self.worker_id]
        end_idx = split_indices[self.worker_id+1]
        with open(self.fname, 'r') as f:
            for line in islice(f, start_idx, end_idx):
                _year, _month, black_elo, white_elo, time_base, time_bonus, result, checkmate, moves = line.split(' ', 8)
                moves = moves.strip().split(' ')
                yield int(black_elo), int(white_elo), float(time_base), float(time_bonus), 4 if result == '3' else int(result), int(checkmate), self.get_indices(moves) if not self.use_pretrained_embedding else self.get_embedding(moves)

    def __len__(self) -> Optional[int]:
        """
        Gets the length of the dataset, or `None` if `total` was not provided upon initialization
        """
        return self.total

class PositionalEncoder(nn.Module):
    """From https://github.com/pytorch/pytorch/issues/51551"""

    def __init__(self, d_model, max_seq_len=SEQUENCE_LENGTH):
        super().__init__()
        self.d_model = d_model
        self.d_model_sqrt = np.sqrt(d_model)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = x * self.d_model_sqrt
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x

class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class modelFlatten(nn.Module):
    
    def __init__(self, number_of_moves: int) -> None:
        """
        Initializes the model which flattens the transformer output before passing it through fully connected layers.
        This is slower and appears no better than `modelAvg`.

        Parameters
        ----------
        number_of_moves : int
            The total number of unique moves (for the `Embedding` layer size)
        """
        super().__init__()
        self.embedding = nn.Embedding(number_of_moves+1, EMBEDDING_DIM, padding_idx=0)
        self.positional_encoding = PositionalEncoder(EMBEDDING_DIM, max_seq_len=SEQUENCE_LENGTH)
        # NOTE: This was trained using torch 1.7.1, so `batch_first` does not exist on `TransformerEncoderLayer`;
        # this necessitates the transposes of `moves` and `masks` in `forward`
        encoder_layer = nn.TransformerEncoderLayer(EMBEDDING_DIM, 6, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.others = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQUENCE_LENGTH*EMBEDDING_DIM, 5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
            nn.Dropout(0.1),
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.1),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(0.1),
            nn.Linear(500, 2)
        )
        self.float()
    
    def forward(self, moves: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes the model forward pass

        Parameters
        ----------
        moves : torch.Tensor
            Tensor of the move indices
        masks : torch.Tensor
            Tensor of the masks for `moves`
        
        Returns
        -------
        torch.Tensor :
            The model output
        """
        encoded = torch.transpose(self.positional_encoding.forward(self.embedding.forward(moves)), 0, 1)
        #encoded = torch.transpose(self.positional_encoding.forward(moves), 0, 1)
        return self.others.forward(torch.transpose(self.encoder.forward(encoded, src_key_padding_mask=masks), 0, 1))

class modelAvg(nn.Module):
    
    def __init__(self, number_of_moves: int) -> None:
        """
        Initializes the model which uses a mean down the second dimension (and is therefore much lighter than `modelFlatten`);
        this appears to achieve similar results.

        Parameters
        ----------
        number_of_moves : int
            The total number of unique moves (for the `Embedding` layer size)
        """
        super().__init__()
        self.embedding = nn.Embedding(number_of_moves+1, EMBEDDING_DIM, padding_idx=0)
        self.positional_encoding = PositionalEncoder(EMBEDDING_DIM, max_seq_len=SEQUENCE_LENGTH)
        # NOTE: This was trained using torch 1.7.1, so `batch_first` does not exist on `TransformerEncoderLayer`;
        # this necessitates the transposes of `moves` and `masks` in `forward`
        encoder_layer = nn.TransformerEncoderLayer(EMBEDDING_DIM, 6, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.others = nn.Sequential(
            nn.Linear(SEQUENCE_LENGTH, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 2)
        )
        self.float()
    
    def forward(self, moves: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes the model forward pass

        Parameters
        ----------
        moves : torch.Tensor
            Tensor of the move indices
        masks : torch.Tensor
            Tensor of the masks for `moves`
        
        Returns
        -------
        torch.Tensor :
            The model output
        """
        encoded = torch.transpose(self.positional_encoding.forward(self.embedding.forward(moves)), 0, 1)
        after_transformer = torch.transpose(self.encoder.forward(encoded, src_key_padding_mask=masks), 0, 1)
        return self.others.forward(torch.mean(after_transformer, dim=2))

class modelLinear(nn.Module):

    def __init__(self, number_of_moves: int) -> None:
        """
        Initializes the model; this version includes no transformer or positional encodings

        Parameters
        ----------
        number_of_moves : int
            The total number of unique moves (for the `Embedding` layer size)
        """
        self.model = nn.Sequential(
                nn.Embedding(number_of_moves+1, EMBEDDING_DIM, padding_idx=0),
                nn.Flatten(),
                nn.Linear(EMBEDDING_DIM*SEQUENCE_LENGTH, 7500),
                nn.ReLU(),
                nn.BatchNorm1d(7500),
                nn.Dropout(0.1),
                nn.Linear(7500, 5000),
                nn.ReLU(),
                nn.BatchNorm1d(5000),
                nn.Dropout(0.1),
                nn.Linear(5000, 1000),
                nn.ReLU(),
                nn.BatchNorm1d(1000),
                nn.Dropout(0.1),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.BatchNorm1d(500),
                nn.Dropout(0.075),
                nn.Linear(500, 100),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Dropout(0.05),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.BatchNorm1d(50),
                nn.Dropout(0.05),
                nn.Linear(50, 2)
            )
    
    def forward(self, moves: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass

        Parameters
        ----------
        moves : torch.Tensor
            Tensor of the move indices
        
        Results
        -------
        torch.Tensor :
            The model output
        """
        return self.model.forward(moves)

def calc_lr(step: int, dim_embed: int, warmup_steps: int) -> float:
    """
    Calculates the new learning rate depending on the current `step`, embedding dimension, and number of warmup steps; see https://kikaben.com/transformers-training-details/

    Parameters
    ----------
    step : int
        Which step of the scheduler we're on
    dim_embed : int
        The number of embedding dimensions
    warmup_steps : int
        The number of warmup steps
    
    Returns
    -------
    float :
        The new learning rate
    """
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

class Scheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer: optim.Optimizer, dim_embed: int, warmup_steps: int, last_epoch: int=-1, verbose: bool=False) -> None:
        """
        Learning rate scheduler from here: https://kikaben.com/transformers-training-details/

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer for which the learning rate is being updated
        dim_embed : int
            The number of embedding dimensions
        warmup_steps : int
            The number of warmup steps
        last_epoch : int, default=-1
            The last epoch for the scheduler
        verbose : bool, default=False
            Whether to print updates
        """
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> List[float]:
        """
        Gets the new learning rate

        Returns
        -------
        list :
            List of float learning rates for each parameter group
        """
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

def setup(rank: int, world_size: int) -> None:
    """
    Sets up the distributed training

    Parameters
    ----------
    rank : int
        Distributed training process number
    world_size : int
        Number of distributed training processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup() -> None:
    """
    Destroys the distributed processes
    """
    dist.destroy_process_group()

DataLoader_T = Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
def get_dataloaders(file_dir: str, num_workers: int, worker_id: int, batch_size: int) -> Tuple[DataLoader_T, DataLoader_T, DataLoader_T]:
    """
    Gets the training, validation, and testing dataloaders

    Parameters
    ----------
    file_dir : str
        Path to the directory that contains the training, validation, and testing data
    num_workers : int
        Number of workers (world size)
    worker_id : int
        Worker number (rank)
    batch_size : int
        Batch size to use

    Returns
    -------
    train_dataloader : DataLoader
        DataLoader for the training data
    val_dataloader : DataLoader
        DataLoade for the validation data
    test_dataloader : DataLoader
        DataLoader for the testing data
    """
    train_dataloader = DataLoader(ChessDataset(os.path.join(file_dir, 'train_data.txt'), num_workers, worker_id, indices_file='indices.json', total=2323143793), batch_size=batch_size, pin_memory=False, num_workers=0)
    val_dataloader = DataLoader(ChessDataset(os.path.join(file_dir, 'val_data.txt'), num_workers, worker_id, indices_file='indices.json', total=282773280), batch_size=batch_size, pin_memory=False, num_workers=0)
    test_dataloader = DataLoader(ChessDataset(os.path.join(file_dir, 'test_data.txt'), num_workers, worker_id, indices_file='indices.json', total=282773280), batch_size=batch_size, pin_memory=False, num_workers=0)
    return train_dataloader, val_dataloader, test_dataloader

def run_model(rank: int, world_size: int) -> None:
    """
    Runs the `DistributedDataParallel` model

    Parameters
    ----------
    rank : int
        Distributed training process number
    world_size : int
        Number of distributed training processes
    """
    setup(rank, world_size)
    train_dl, val_dl, test_dl = get_dataloaders(FILE_DIR, num_workers=world_size, worker_id=rank, batch_size=128 if rank == 0 else 256)

    m = modelAvg().to(rank)
    net = DistributedDataParallel(m, device_ids=[rank], find_unused_parameters=True)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    scheduler = Scheduler(optimizer, EMBEDDING_DIM, int(len(train_dl)/2))

    i = 0
    for epoch in range(5):
        losses: List[Tuple[Number, Number, Number]] = []
        net.train()
        with tqdm(total=len(train_dl), desc='Training epoch {}'.format(epoch+1), position=rank) as pbar:
            for black_elo, white_elo, time_base, time_bonus, result, checkmate, (moves, masks) in train_dl:
                i += 1
                # if i < 1100000:
                #     continue
                optimizer.zero_grad()
                elos = torch.stack((black_elo, white_elo), dim=1).to(rank, dtype=torch.float32)
                outputs = net.forward(moves.to(rank), masks.to(rank))
                loss: torch.Tensor = criterion(outputs, elos)
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                if i % 100 == 0:
                    with torch.no_grad():
                        black_diff = torch.mean(torch.abs(elos[:, 0] - outputs[:, 0])).item()
                        white_diff = torch.mean(torch.abs(elos[:, 1] - outputs[:, 1])).item()
                        # std_dev_black = outputs[:, 0].std().item()
                        # std_dev_white = outputs[:, 1].std().item()
                        # std_dev = (std_dev_black + std_dev_white)/2
                        l = loss.item()
                        losses.append((l, black_diff, white_diff))
                        pbar.set_postfix_str('Loss: {:.1f} | MAE black: {:.1f}, white: {:.1f}'.format(l, black_diff, white_diff))
                if i != 0 and i % 100000 == 0:
                    if rank == 0:
                        print('Saved checkpoint', i//100000)
                        torch.save(net.state_dict(), os.path.join('checkpoints', 'checkpoint-ddp.pth'))
                        with open(os.path.join('checkpoints', 'checkpoint-{}_stats.pkl'.format(i//100000)), 'wb') as f:
                            pickle.dump({'loss': [l[0] for l in losses], 'black_diff': [l[1] for l in losses], 'white_diff': [l[2] for l in losses]}, f)
                    dist.barrier()
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                    net.load_state_dict(torch.load(os.path.join('checkpoints', 'checkpoint-ddp.pth'), map_location=map_location))

        if rank == 0:
            torch.save(net.state_dict(), 'model-ddp.pth'.join(epoch+1))
            with open('train_stats_epoch_{}.pkl'.format(epoch+1), 'wb') as f:
                pickle.dump({'loss': [l[0] for l in losses], 'black_diff': [l[1] for l in losses], 'white_diff': [l[2] for l in losses]}, f)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        net.load_state_dict(torch.load('model-ddp.pth', map_location=map_location))
        # net.eval()
        # if rank == 0:
        #     black_diffs: List[torch.float32] = []
        #     white_diffs: List[torch.float32] = []
        #     with torch.no_grad():
        #         for black_elo, white_elo, time_base, time_bonus, result, checkmate, (moves, masks) in tqdm(val_dl, desc='Validating'):
        #             elos = torch.stack((black_elo, white_elo), dim=1).to(DEVICE, dtype=torch.float32)
        #             outputs = net.forward(moves.to(DEVICE), masks.to(DEVICE))
        #             black_diffs.extend(torch.abs(elos[:, 0] - outputs[:, 0]).cpu().numpy())
        #             white_diffs.extend(torch.abs(elos[:, 1] - outputs[:, 1]).cpu().numpy())
        #     with open('val_stats_epoch_{}.pkl'.format(epoch+1), 'wb') as f:
        #         pickle.dump({'black_diffs': black_diffs, 'white_diffs': white_diffs}, f)
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(run_model, args=(world_size,), nprocs=world_size, join=True)
