import os, pickle, torch
import matplotlib.pyplot as plt
import numpy as np
from train import modelAvg, modelTime, modelLinear, get_dataloaders
from collections import OrderedDict
from itertools import islice
from tqdm import tqdm
from scipy.stats import gaussian_kde

if not os.path.exists('images'):
    os.mkdir('images')

n = max(int(fname[11:-10]) for fname in os.listdir('checkpoints') if fname.startswith('checkpoint-') and fname.endswith('_stats.pkl'))

with open(os.path.join('checkpoints', 'checkpoint-{}_stats.pkl'.format(n)), 'rb') as f:
    data = pickle.load(f)

plt.plot(data['loss'])
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(os.path.join('images', 'loss.png'))
plt.clf()

plt.plot(data['white_diff'])
plt.plot(data['black_diff'])
plt.yscale('log')
plt.ylim(bottom=min(75, min(data['white_diff']), min(data['black_diff'])))
plt.xlabel('Iteration')
plt.ylabel('Average absolute Elo diff')
plt.tight_layout()
plt.savefig(os.path.join('images', 'diffs.png'))

_, val_dl, __ = get_dataloaders('/mnt/csd_gpu/data')
net = modelTime().to('cpu')

state_dict = torch.load(os.path.join('checkpoints', 'checkpoint-{}.pth'.format(n)), map_location='cpu')
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    k = k.replace('module.', '')
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)
net.eval()
black_diffs = []
white_diffs = []
black_elos = []
white_elos = []
black_preds = []
white_preds = []
with torch.no_grad():
    for black_elo, white_elo, time_base, time_bonus, result, checkmate, (moves, masks) in tqdm(islice(val_dl, 25), total=25):
        elos = torch.stack((black_elo, white_elo), dim=1).to(dtype=torch.float32)
        outputs = net.forward(moves, masks, time_base.to(dtype=torch.float32), time_bonus.to(dtype=torch.float32))
        black_diff = torch.abs(elos[:, 0] - outputs[:, 0])
        white_diff = torch.abs(elos[:, 1] - outputs[:, 1])
        black_diffs.extend(black_diff.numpy())
        white_diffs.extend(white_diff.numpy())
        black_elos.extend(black_elo.numpy())
        white_elos.extend(white_elo.numpy())
        black_preds.extend(outputs[:, 0].numpy())
        white_preds.extend(outputs[:, 1].numpy())

black_diffs = np.array(black_diffs)
white_diffs = np.array(white_diffs)
black_elos = np.array(black_elos)
white_elos = np.array(white_elos)
black_preds = np.array(black_preds)
white_preds = np.array(white_preds)

print('Average black difference:', black_diffs.mean(), '±', black_diffs.std())
print('Average white differences:', white_diffs.mean(), '±', white_diffs.std())

black_under = np.empty(2500)
white_under = np.empty(2500)
for d in range(2500):
    black_under[d] = 100*np.count_nonzero(black_diffs <= d)/black_diffs.shape[0]
    white_under[d] = 100*np.count_nonzero(white_diffs <= d)/white_diffs.shape[0]
plt.clf()
plt.plot(range(2500), black_under, '-o')
plt.plot(range(2500), white_under, '-o')
plt.tight_layout()
plt.savefig(os.path.join('images', 'under.png'))

black_percentile = np.percentile(black_diffs, np.arange(100))
white_percentile = np.percentile(white_diffs, np.arange(100))
plt.clf()
plt.plot(np.arange(100), black_percentile, '-o')
plt.plot(np.arange(100), white_percentile, '-o')
plt.xlabel('Percentile')
plt.ylabel('Absolute Elo diff')
plt.tight_layout()
plt.savefig(os.path.join('images', 'percentiles.png'))

black_buckets = {}
white_buckets = {}
for black_elo, black_elo_diff, white_elo, white_elo_diff in zip(black_elos, black_diffs, white_elos, white_diffs):
    black_elo_bucket = round(black_elo, -2)
    white_elo_bucket = round(white_elo, -2)
    try:
        black_buckets[black_elo_bucket].append(black_elo_diff)
    except KeyError:
        black_buckets[black_elo_bucket] = [black_elo_diff]
    try:
        white_buckets[white_elo_bucket].append(white_elo_diff)
    except KeyError:
        white_buckets[white_elo_bucket] = [white_elo_diff]

black_keys = sorted(black_buckets.keys())
white_keys = sorted(white_buckets.keys())
plt.clf()
plt.plot(black_keys, [np.mean(black_buckets[k]) for k in black_keys], '-o')
plt.plot(white_keys, [np.mean(white_buckets[k]) for k in white_keys], '-o')
plt.xlabel('Actual Elo')
plt.ylabel('Average absolute Elo diff')
plt.tight_layout()
plt.savefig(os.path.join('images', 'bucketed.png'))

plt.clf()
plt.hist(black_diffs, bins=50)
plt.xlabel('Absolute Elo diff')
plt.ylabel('Number of games')
plt.tight_layout()
plt.savefig(os.path.join('images', 'black_diffs.png'))
plt.clf()
plt.hist(white_diffs, bins=50)
plt.xlabel('Absolute Elo diff')
plt.ylabel('Number of games')
plt.tight_layout()
plt.savefig(os.path.join('images', 'white_diffs.png'))

print('Number of points:', len(black_elos))
xy = np.vstack([black_elos, black_preds])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = black_elos[idx], black_preds[idx], z[idx]
plt.clf()
plt.scatter(x, y, c=z)
plt.xlabel('Actual Elo')
plt.ylabel('Predicted Elo')
plt.tight_layout()
plt.savefig(os.path.join('images', 'actual_vs_predicted_black.png'))
xy = np.vstack([white_elos, white_preds])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = white_elos[idx], white_preds[idx], z[idx]
plt.clf()
plt.scatter(x, y, c=z)
plt.xlabel('Actual Elo')
plt.ylabel('Predicted Elo')
plt.tight_layout()
plt.savefig(os.path.join('images', 'actual_vs_predicted_white.png'))
