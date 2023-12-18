import json, pickle
import cupy as cp
import numpy as np
from tqdm import tqdm
from typing import List

print('-'*5, 'AVERAGE', '-'*5)
with open('avg_stats.pkl', 'rb') as f:
    data = pickle.load(f)

def get_results(prediction: int, preds: cp.ndarray, actuals: cp.ndarray) -> List[cp.ndarray]:
    """
    Gets various stats about the given data

    Parameters
    ----------
    prediction : int
        The predicted Elo
    preds : cp.ndarray
        Cupy array of all predicted Elos
    actuals : cp.ndarray
        Cupy array of the actual Elos corresponding to `preds`
    
    Returns
    -------
    to_return : list
        If `actual_at_pred` is an array of the actual Elos wherever `preds` equals `prediction`:
        * Number of entries in `pred` that equal `prediction`
        * Minimum of `actual_at_pred`
        * Maximum of `actual_at_pred`
        * Mean of `actual_at_pred`
        * Standard deviation of `actual_at_pred`
        * Median of `actual_at_pred`
        * Percentiles of `actual_at_pred` at 25, 75, 10, 90, 5, 95, 0.5, and 99.5
    """
    mask = preds == prediction
    actual_at_pred = actuals[mask]
    to_return = [cp.count_nonzero(mask), actual_at_pred.min(), actual_at_pred.max(), actual_at_pred.mean(), cp.std(actual_at_pred), cp.median(actual_at_pred)]
    for p in (25.0, 75.0, 10.0, 90.0, 5.0, 95.0, 0.5, 99.5):
        to_return.append(cp.percentile(actual_at_pred, p))
    return to_return

for k, v in data.items():
    data[k] = cp.array(v).astype(np.int32)

print('MAE white:', cp.mean(cp.abs(data['actual_white_elos'] - data['pred_white_elos'])))
print('MAE black:', cp.mean(cp.abs(data['actual_black_elos'] - data['pred_black_elos'])))
preds_to_actual_black = {}
for pred_black_elo in tqdm(cp.unique(data['pred_black_elos'])):
    results = get_results(pred_black_elo, data['pred_black_elos'], data['actual_black_elos'])
    d = {}
    for i, n in enumerate(('number', 'min', 'max', 'mean', 'stddev', 'median', '25th', '75th', '10th', '90th', '5th', '95th', '0.5th', '99.5th')):
        d[n] = int(results[i])
    preds_to_actual_black[int(pred_black_elo.item())] = d

with open('avg_percentiles_black.json', 'w') as f:
    json.dump(preds_to_actual_black, f)

preds_to_actual_white = {}
for pred_white_elo in tqdm(cp.unique(data['pred_white_elos'])):
    results = get_results(pred_white_elo, data['pred_white_elos'], data['actual_white_elos'])
    d = {}
    for i, n in enumerate(('number', 'min', 'max', 'mean', 'stddev', 'median', '25th', '75th', '10th', '90th', '5th', '95th', '0.5th', '99.5th')):
        d[n] = int(results[i])
    preds_to_actual_white[int(pred_white_elo.item())] = d

with open('avg_percentiles_white.json', 'w') as f:
    json.dump(preds_to_actual_white, f)

print('-'*5, 'TIME', '-'*5)
with open('avg_time_stats.pkl', 'rb') as f:
    data = pickle.load(f)
tmp = data['pred_white_elos']
data['pred_white_elos'] = data['pred_black_elos']
data['pred_black_elos'] = tmp

for k, v in data.items():
    data[k] = cp.array(v).astype(np.int32)

print('MAE white:', cp.mean(cp.abs(data['actual_white_elos'] - data['pred_white_elos'])))
print('MAE black:', cp.mean(cp.abs(data['actual_black_elos'] - data['pred_black_elos'])))
preds_to_actual_black = {}
for pred_black_elo in tqdm(cp.unique(data['pred_black_elos'])):
    #mask = data['pred_black_elos'] == pred_black_elo
    #actual_at_pred = data['actual_black_elos'][mask]
    results = get_results(pred_black_elo, data['pred_black_elos'], data['actual_black_elos'])
    d = {}
    for i, n in enumerate(('number', 'min', 'max', 'mean', 'stddev', 'median', '25th', '75th', '10th', '90th', '5th', '95th', '0.5th', '99.5th')):
        d[n] = int(results[i])
    preds_to_actual_black[int(pred_black_elo.item())] = d

with open('avg_time_percentiles_black.json', 'w') as f:
    json.dump(preds_to_actual_black, f)

preds_to_actual_white = {}
for pred_white_elo in tqdm(cp.unique(data['pred_white_elos'])):
    results = get_results(pred_white_elo, data['pred_white_elos'], data['actual_white_elos'])
    d = {}
    for i, n in enumerate(('number', 'min', 'max', 'mean', 'stddev', 'median', '25th', '75th', '10th', '90th', '5th', '95th', '0.5th', '99.5th')):
        d[n] = int(results[i])
    preds_to_actual_white[int(pred_white_elo.item())] = d

with open('avg_time_percentiles_white.json', 'w') as f:
    json.dump(preds_to_actual_white, f)
