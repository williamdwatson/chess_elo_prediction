import json, os
from statistics import mean, stdev
from tqdm import tqdm
from typing import Dict, List, Tuple
from settings import FILE_DIR

YEAR1 = '2015'
YEAR2 = '2022'

# Check within the same year
move_scores1: Dict[int, List[Tuple[int, int]]] = {}
move_scores2: Dict[int, List[Tuple[int, int]]] = {}
files = [os.path.join(FILE_DIR, n) for n in ('train_data.txt', 'val_data.txt', 'test_data.txt')]
for fname in files:
    with open(fname, 'r') as f:
        for line in tqdm(f, desc='Parsing {}'.format(os.path.basename(fname))):
            split = line.split(' ', 8)
            if split[0].strip() == YEAR1:
                try:
                    move_scores1[hash(split[-1])].append((int(split[2]), int(split[3])))
                except KeyError:
                    move_scores1[hash(split[-1])] = [(int(split[2]), int(split[3]))]
            elif split[0].strip() == YEAR2:
                try:
                    move_scores2[hash(split[-1])].append((int(split[2]), int(split[3])))
                except KeyError:
                    move_scores2[hash(split[-1])] = [(int(split[2]), int(split[3]))]

black_avgs1 = []
black_stds1 = []
black_avgs2 = []
black_stds2 = []
white_avgs1 = []
white_stds1 = []
white_avgs2 = []
white_stds2 = []
for game_hash, elos1 in move_scores1.items():
    try:
        elos2 = move_scores2[game_hash]
        assert len(elos1) > 0 and len(elos2) > 0
        black_elos1 = [e[0] for e in elos1]
        white_elos1 = [e[1] for e in elos1]
        black_elos2 = [e[0] for e in elos2]
        white_elos2 = [e[1] for e in elos2]
        black_avgs1.append(mean(black_elos1))
        black_stds1.append(stdev(black_elos1) if len(black_elos1) > 1 else 0)
        black_avgs2.append(mean(black_elos2))
        black_stds2.append(stdev(black_elos2) if len(black_elos2) > 1 else 0)
        white_avgs1.append(mean(white_elos1))
        white_stds1.append(stdev(white_elos1) if len(white_elos1) > 1 else 0)
        white_avgs2.append(mean(white_elos2))
        white_stds2.append(stdev(white_elos2) if len(white_elos2) > 1 else 0)
    except KeyError:
        continue

with open('same_game_stats.json', 'w') as f:
    json.dump({
            'black_avgs1': black_avgs1, 'black_stds1': black_stds1, 'black_avgs2': black_avgs2, 'black_stds2': black_stds2,
            'white_avgs1': white_avgs1, 'white_stds1': white_stds1, 'white_avgs2': white_avgs2, 'white_stds2': white_stds2
        }, f)
