import scipy.io as sio
import os
import shutil
import numpy as np

# load database info
live_data = sio.loadmat('LIVEVIDEOData.mat')
file_list = live_data['file_name']
ref_list = live_data['ref_name']
ref_no = live_data['ref_no']

# define database path
database_path = r'E:\DATABASE\videodatabase\live\videos'

# tempporary file path, no slash!
temp = r'temp'
result, data = os.path.join(temp, 'result'), os.path.join(temp, 'data')

# three output array: spatial score, temporal score, overall score
s_movie, t_movie, movie = [], [], []

# run on database
for i in range(0, len(file_list)):
    # delete last temporary files
    if os.path.exists(temp):
        shutil.rmtree(temp)

    os.makedirs(result)
    os.makedirs(data)

    # define reference video name and distorted video name
    ref_name = os.path.join(database_path, ref_list[0][ref_no[i, 0] - 1][0])
    dis_name = os.path.join(database_path, file_list[i][0][0])

    # construct command line
    cmd = 'movie ' + ref_name + ' ' + dis_name + ' ' + 'ref ' + 'dis ' + result + ' ' + data + ' ' + '768 ' + '432 ' + '-f ' + 'temp/result ' + '-remove'
    print('{}: {}'.format(i, cmd))

    # run movie
    os.system(cmd)

    # save scores
    with open(os.path.join(data, 'dis_movie.txt'), 'r') as f:
        movie.append(float(f.readline()))
    with open(os.path.join(data, 'dis_smovie.txt'), 'r') as f:
        s_movie.append(float(f.readline()))
    with open(os.path.join(data, 'dis_tmovie.txt'), 'r') as f:
        t_movie.append(float(f.readline()))

    s = np.array(s_movie, dtype=np.float32)
    t = np.array(t_movie, dtype=np.float32)
    a = np.array(movie, dtype=np.float32)

    sio.savemat('score_30.mat', {'s': s, 't': t, 'a': a})    
