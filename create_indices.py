import json, os
from itertools import chain
from tqdm import tqdm
from settings import FILE_DIR

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
    with open('indices.json', 'w') as f:
        json.dump({mv: i for i, mv in enumerate(all_moves)}, f)
    with open('lengths.json', 'w') as f:
        json.dump({'train': lengths[0], 'val': lengths[1], 'test': lengths[2]}, f)
    print('Number of unique moves:', len(all_moves))
    print('Characters present in moves:', set(chain(*all_moves)))
