from collections import OrderedDict
import pickle, torch
import torch.nn as nn
import numpy as np
from train import modelAvg, modelTime, get_dataloaders
from tqdm import tqdm
from settings import FILE_DIR

DEVICE = torch.device('cuda')

_, val_dl, __ = get_dataloaders(FILE_DIR)

net = nn.DataParallel(modelAvg())
net.load_state_dict(torch.load('model_avg.pth'))
net.cuda()
net.eval()

actual_white_elos = np.empty(282773280, dtype=np.int32)
actual_black_elos = np.empty_like(actual_white_elos)
pred_white_elos = np.empty_like(actual_white_elos)
pred_black_elos = np.empty_like(actual_black_elos)

i = 0
with torch.no_grad():
    for black_elo, white_elo, time_base, time_bonus, result, checkmate, (moves, masks) in tqdm(val_dl):
        outputs = net.forward(moves.to(DEVICE), masks.to(DEVICE))
        s = slice(i, i+white_elo.shape[0])
        actual_white_elos[s] = white_elo.cpu()
        actual_black_elos[s] = black_elo.cpu()
        pred_white_elos[s] = outputs[:, 1].cpu()
        pred_black_elos[s] = outputs[:, 0].cpu()
        i += white_elo.shape[0]

with open('avg_stats.pkl', 'wb') as f:
    pickle.dump({
        'actual_white_elos': actual_white_elos,
        'actual_black_elos': actual_black_elos,
        'pred_white_elos': pred_white_elos,
        'pred_black_elos': pred_black_elos
    }, f)

pred_white_elos.fill(0)
pred_black_elos.fill(0)

net = nn.DataParallel(modelTime())
net.load_state_dict(torch.load('model_avg_time.pth'))
net.cuda()
net.eval()

i = 0
with torch.no_grad():
    for black_elo, white_elo, time_base, time_bonus, result, checkmate, (moves, masks) in tqdm(val_dl):
        outputs = net.forward(moves.to(DEVICE), masks.to(DEVICE), time_base.to(DEVICE, dtype=torch.float32), time_bonus.to(DEVICE, dtype=torch.float32))
        s = slice(i, i+white_elo.shape[0])
        actual_white_elos[s] = white_elo.cpu()
        actual_black_elos[s] = black_elo.cpu()
        pred_white_elos[s] = outputs[:, 1].cpu()
        pred_black_elos[s] = outputs[:, 0].cpu()
        i += white_elo.shape[0]

with open('avg_time_stats.pkl', 'wb') as f:
    pickle.dump({
        'actual_white_elos': actual_white_elos,
        'actual_black_elos': actual_black_elos,
        'pred_white_elos': pred_white_elos,
        'pred_black_elos': pred_black_elos
    }, f)
