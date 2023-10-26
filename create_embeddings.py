import json, os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
from typing import Dict, Generator, List, Optional
from settings import FILE_DIR

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model: Word2Vec):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

class FileStreamer:

    def __init__(self, files_with_counts: Dict[str, Optional[int]]) -> None:
        """
        Streams moves from the downloaded data

        Parameters
        ----------
        files_with_counts : dict
            Dictionary mapping string filenames to optional lengths
        """
        self.files_with_counts = files_with_counts

    def __iter__(self) -> Generator[List[str], None, None]:
        """
        Loops through the supplied files

        Yields
        ------
        list :
            List of move strings
        """
        for filename, line_count in self.files_with_counts.items():
            with open(filename, 'r') as f:
                for i, line in enumerate(tqdm(map(str.strip, f), total=line_count, desc=os.path.basename(filename))):
                    if line == '':
                        print('Bad line in {}: {}'.format(filename, i))
                        continue
                    yield line.split(' ')[8:]   # The first 8 items are year, month, elos, times, etc.

if __name__ == '__main__':
    files = [os.path.join(FILE_DIR, n) for n in ('train_data.txt', 'val_data.txt', 'test_data.txt')]
    files_with_counts: Dict[str, int] = {}
    all_moves = set()
    for fname in files:
        with open(fname, 'r') as f:
            num = 0
            for line in tqdm(f, desc='Getting line count of {}'.format(os.path.basename(fname))):
                all_moves.update(map(str.strip, line.split(' ')[8:]))
                num += 1
            files_with_counts[fname] = num
    print('All unique moves:', all_moves)
    print('Total number of unique moves:', len(all_moves))
    model = Word2Vec(
        FileStreamer(files_with_counts),
        vector_size=72,
        window=3,
        min_count=1,
        workers=6,
        callbacks=[callback()]
    )
    model.save('embeddings.model')
    
    move_vecs = model.wv
    move_vecs.save('move_vecs.wordvectors')

    with open('lengths.json', 'w') as f:
        json.dump({
            'train': files_with_counts['train_data.txt'],
            'val': files_with_counts['val_data.txt'],
            'test': files_with_counts['test_data.txt']
        }, f)
