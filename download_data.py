import glob, itertools, os, re, shutil, tempfile
from settings import FILE_DIR, SEQUENCE_LENGTH
tempfile.tempdir = FILE_DIR
import multiprocessing as mp
from typing import List, Optional
from pyzstd import decompress_stream
from tqdm import tqdm
from urllib.request import urlretrieve, urlcleanup

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
        Whether `mv` contains any banned characters or has fewer than two characters
    """
    return len(mv) > 1 and not any(bad in mv for bad in ('.', '[', ']', '{', '}', '%', ':'))

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
    return mv.replace('!', '').replace('?', '').replace('+', '').replace(',', '').strip()

def parse_file(link: str, save_file_path: str, process_num: Optional[int]=None, min_time_base: int=180, max_time_base: int=600, min_time_bonus: int=0, max_time_bonus: int=100, min_game_length: int=12, max_game_length: int=SEQUENCE_LENGTH) -> int:
    """
    Downloads and parses a file

    Parameters
    ----------
    link : str
        Link to the data to download and parse
    save_file_path : str
        Filepath to the master data file to append to
    process_num : int, optional, default=None
        Which process number the function is being run in (for positioning of the progress display)
    min_time_base : int, default=180
        Minimum base time (in seconds, inclusive) for a game to be accepted
    max_time_base : int, default=600
        Maximum base time (in seconds, inclusive) for a game to be accepted
    min_time_bonus : int, default=0
        Minimum bonus time (in seconds, inclusive) for a game to be accepted
    max_time_bonus : int, default=100
        Maximum bonus time (in seconds, inclusive) for a game to be accepted
    min_game_length : int, default=12
        Minimum game length (in plys, inclusive) for a game to be accepted
    max_game_length : int, default=SEQUENCE_LENGTH
        Maximum game length (in plys, inclusive) for a game to be accepted; this parameter is important as it is used in the size of the transformer later
    
    Returns
    -------
    num_games : int
        The number of games that passed the cutoffs and were written to the file
    """
    COMMENT_PATTERN = re.compile('{.*?}')
    basename = os.path.basename(link)
    year = int(basename.replace('lichess_db_standard_rated_', '').split('-')[0])
    month = int(basename.replace('lichess_db_standard_rated_', '').split('-')[1].split('.')[0])
    num_games = 0
    with tempfile.TemporaryFile() as decompressed_tmp:
        # Download the file and decompress it
        download_tmp, headers = urlretrieve(link)
        print('Downloaded', link)
        with open(download_tmp, 'rb') as downloaded_f:
            decompress_stream(downloaded_f, decompressed_tmp)
        print('Decompressed', download_tmp)
        urlcleanup()
        move_list = []
        black_elo_list = []
        white_elo_list = []
        time_base_list = []
        time_bonus_list = []
        result_list = []
        checkmate_list = []
        current_moves = None
        current_black_elo = None
        current_white_elo = None
        time_base = None
        time_bonus = None
        result = None
        checkmate = False
        decompressed_tmp.seek(0)
        # Loop through each line in the bytes file
        for line in tqdm(decompressed_tmp, desc=basename, position=process_num):
            # Convert the line to text
            line = line.decode('utf-8')
            assert not line.startswith('2.')
            # If we're at an event, we should have just finished the previous one; check if we got everything we need
            if line.startswith('[Event'):
                # If we have what we need within the set contraints, save that data
                if all(x is not None for x in (current_moves, current_black_elo, current_white_elo, time_base, time_bonus, result))\
                    and min_time_base <= time_base <= max_time_base and min_time_bonus <= time_bonus <= max_time_bonus and min_game_length <= len(current_moves) <= max_game_length:
                    move_list.append(current_moves)
                    black_elo_list.append(current_black_elo)
                    white_elo_list.append(current_white_elo)
                    time_base_list.append(time_base)
                    time_bonus_list.append(time_bonus)
                    result_list.append(result)
                    checkmate_list.append(checkmate)
                    num_games += 1
                    # Write the data file in chunks
                    if len(move_list) > 500000:
                        to_write = '\n'.join('{} {} {} {} {} {} {} {} {}'.format(year, month, black_elo, white_elo, t1, t2, r, int(c), ' '.join(moves)) for moves, black_elo, white_elo, t1, t2, r, c in zip(move_list, black_elo_list, white_elo_list, time_base_list, time_bonus_list, result_list, checkmate_list))
                        with open(save_file_path, 'a') as f:
                            f.write(to_write + '\n')
                        move_list.clear()
                        black_elo_list.clear()
                        white_elo_list.clear()
                        time_base_list.clear()
                        time_bonus_list.clear()
                        result_list.clear()
                        checkmate_list.clear()
                current_moves = None
                current_black_elo = None
                current_white_elo = None
                time_base = None
                time_bonus = None
                result = None
                checkmate = False
            # Grab the white Elo as an integer
            elif line.startswith('[WhiteElo'):
                try:
                    current_white_elo = int(line.split('"')[1])
                except ValueError:
                    pass
            # Grab the black Elo as an integer
            elif line.startswith('[BlackElo'):
                try:
                    current_black_elo = int(line.split('"')[1])
                except ValueError:
                    pass
            # This should be the move line
            elif line.startswith('1.'):
                try:
                    # Strip away unnecessary characters (like "??") and remove bad parts (like evaluation comments)
                    current_moves = [prune_to_move(move).strip() for move in COMMENT_PATTERN.sub('', line.strip()).split(' ')[:-1] if check_move_only(move)]
                    if '#' in current_moves[-1]:
                        checkmate = True
                        current_moves[-1] = current_moves[-1].replace('#', '')
                except Exception:
                    pass
            # Grab the time base and bonus as numbers
            elif line.startswith('[TimeControl'):
                t = line.split('"')[1]
                try:
                    tbase, tbonus = t.split('+')
                    time_base = int(tbase)
                    time_bonus = int(tbonus)
                except Exception:
                    pass
            # Grab who won
            elif line.startswith('[Result'):
                if line.strip() == '1-0':
                    result = 1
                elif line.strip() == '0-1':
                    result = 2
                else:
                    result = 3
        # If we finish the loop with data still present, write it
        if len(move_list) > 0:
            to_write = '\n'.join('{} {} {} {} {} {} {} {} {}'.format(year, month, black_elo, white_elo, t1, t2, r, int(c), ' '.join(moves)) for moves, black_elo, white_elo, t1, t2, r, c in zip(move_list, black_elo_list, white_elo_list, time_base_list, time_bonus_list, result_list, checkmate_list))
            with open(save_file_path, 'a') as f:
                f.write(to_write + '\n')
        return num_games

def process_chunk(process_num: int, links: List[str], dir_name: str) -> None:
    """
    Processes a chunk of files

    Parameters
    ----------
    process_num : int
        Which number process this is
    links : list
        List of string links to process
    """
    for link in map(str.strip, links):
        print('Processing', link)
        num_saved = parse_file(link, os.path.join(dir_name, 'data{}.txt'.format(process_num)), process_num=process_num)
        print('{} games saved for {}'.format(num_saved, os.path.basename(link)))

if __name__ == '__main__':
    mp.freeze_support()
    # Read in the links to the games to grab
    with open('game_list.txt', 'r') as f:
        links = f.readlines()
    links.reverse()
    # Prepare to run multiple processes; if NUM_PROCESSES is too big you may run into rate limits
    NUM_PROCESSES = 5
    link_chunks = [links[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)]
    processes: List[mp.Process] = []
    # Start the chunked processing
    for i, l in enumerate(link_chunks):
        processes.append(mp.Process(target=process_chunk, args=(i, l, FILE_DIR)))
        processes[-1].start()
    for p in processes:
        p.join()
    # The downloaded files should be more or less mixed between the actual files
    # Use the first file as the val/test data
    with open(os.path.join(FILE_DIR, 'data0.txt'), 'r') as f,\
         open(os.path.join(FILE_DIR, 'val_data.txt'), 'w') as val_file,\
         open(os.path.join(FILE_DIR, 'test_data.txt'), 'w') as test_file:
        length = sum(1 for _ in tqdm(f, desc='Checking validation/test file length'))
        f.seek(0)
        val_length = int(length/2)
        test_length = length - val_length
        val_file.writelines(itertools.islice(tqdm(f, desc='Writing validation data', total=val_length), val_length))
        test_file.writelines(tqdm(f, desc='Writing testing data', total=test_length))
    os.remove(os.path.join(FILE_DIR, 'data0.txt'))
    # Combine the other files; see https://stackoverflow.com/a/27077437
    with open(os.path.join(FILE_DIR, 'train_data.txt'), 'wb') as wfd:
        for f in glob.glob(os.path.join(FILE_DIR, 'data*.txt'), recursive=False):
            if 'data0.txt' not in f:
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                os.remove(f)
