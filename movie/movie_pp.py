from multiprocessing import cpu_count, Process, Pool
import scipy.io as sio
import os
import shutil

def movie(ref_name, dis_name):
    pre_name = os.path.basename(dis_name).split('.')[0]
    temp = os.path.join('temp', pre_name)
    result, data = os.path.join(temp, 'result'), os.path.join(temp, 'data')
    os.makedirs(temp)
    os.makedirs(result)
    os.makedirs(data)
    cmd = 'movie ' + ref_name + ' ' + dis_name + ' ' + 'ref ' + 'dis ' + result + ' ' + data + ' ' + '768 ' + '432 ' + '-f ' + result + ' ' + '-remove'
    os.system(cmd)

    save_path = os.path.join('result', pre_name)
    os.mkdir(save_path)
    shutil.copyfile(os.path.join(data, 'dis_movie.txt'), os.path.join(save_path, 'dis_movie.txt'))
    shutil.copyfile(os.path.join(data, 'dis_smovie.txt'), os.path.join(save_path, 'dis_smovie.txt'))
    shutil.copyfile(os.path.join(data, 'dis_tmovie.txt'), os.path.join(save_path, 'dis_tmovie.txt'))

    shutil.rmtree(temp)

if __name__ == '__main__':

    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists('temp'):
        os.mkdir('temp')

    ncpus = cpu_count()
    jobs = []

    live_data = sio.loadmat('LIVEVIDEOData.mat')
    file_list = live_data['file_name']
    ref_list = live_data['ref_name']
    ref_no = live_data['ref_no']
    database_path = r'E:\DATABASE\videodatabase\live\videos'

    for i in range(80, len(file_list)):
        ref_name = os.path.join(database_path, ref_list[0][ref_no[i, 0] - 1][0])
        dis_name = os.path.join(database_path, file_list[i][0][0])
        jobs.append(Process(target=movie, args=(ref_name, dis_name)))
        jobs[-1].start()
        if len(jobs) == ncpus:
            for j in jobs:
                j.join()
            jobs.clear()
        print(i)
