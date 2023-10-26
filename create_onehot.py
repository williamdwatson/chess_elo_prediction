import json, os
from itertools import chain
from tqdm import tqdm
from typing import List
from settings import FILE_DIR

def get_zeros_except(idx: int, length: int) -> List[int]:
    """
    Gets a list of zeros, except at index `idx` (where the value will be 1)

    Parameters
    ----------
    idx : int
        Index to set to 1
    length : int
        Length of the returned list
    
    Returns
    -------
    l : list
        List of `length` zeros, except for a 1 at `idx`
    """
    l = [0]*length
    l[idx] = 1
    return l

if __name__ == '__main__':
    all_moves = set()
    files = [os.path.join(FILE_DIR, n) for n in ('train_data.txt', 'val_data.txt', 'test_data.txt')]
    lengths = []
    for fname in files:
        length = 0
        with open(fname, 'r') as f:
            for line in tqdm(f, desc='Getting moves from {}'.format(os.path.basename(fname))):
                all_moves.update(map(str.strip, line.split(' ')[8:]))
                length += 1
        lengths.append(length)
    needed_len = len(all_moves)
    onehot = {mv: get_zeros_except(i, len(all_moves)) for i, mv in enumerate(all_moves)}
    with open('onehot.json', 'w') as f:
        json.dump(onehot, f)
    with open('lengths.json', 'w') as f:
        json.dump({'train': lengths[0], 'val': lengths[1], 'test': lengths[2]}, f)
    print('Number of unique moves:', len(onehot))
    print('Characters present in moves:', set(chain(*all_moves)))
