import os
from pydub import AudioSegment

path = "data_process/CTTsegment_remove_30"

audiolist = []

for f in os.listdir(path):
    filepath = path + '/' + f
    if os.path.isfile(filepath) and f.find("wav") != -1:
        # # with segment
        # file = f.split('\\')[-1].split('_')[0]
        # without segment
        file = f.split('\\')[-1].split('.')[0].split('_')[0]
        print(file)
        filepath = filepath.replace("/", "\\")
        idx1 = 1
        for f in range(5):
            newpath = "data_process/CTT5-" + str(idx1) + "/" + file + ".wav"
            if os.path.isfile(newpath):
                break
            else:
                idx1 += 1
        cmd1 = "copy " + filepath + " D:\\Lulu\\Research\\0709\\LAS_Mandarin_PyTorch-master\\data_process\\CTT5-" + str(idx1) + "_10"
        # print(cmd1)
        # os.system(cmd1)
        idx2 = 1
        for f in range(5):
            newpath = "data_process/CTT5-" + str(idx2) + "-2/" + file + ".wav"
            if os.path.isfile(newpath):
                break
            else:
                idx2 += 1
        if(idx2 < 6):
            cmd2 = "copy " + filepath + " D:\\Lulu\\Research\\0709\\LAS_Mandarin_PyTorch-master\\data_process\\CTT5-" + str(idx2) + "-19"
            print(newpath)
            print(cmd2)
            os.system(cmd2)
