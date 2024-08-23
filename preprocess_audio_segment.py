import os
from pydub import AudioSegment

path = "data_process/Lu_CTTdeletion_d"
path2 = "data_process/Lu_CTTdeletion_o_d"

audiolist = []

for f in os.listdir(path):
    filepath = path + '/' + f
    if os.path.isfile(filepath) and f.find("wav") != -1:
        sound = AudioSegment.from_wav(filepath)
        idx = 1
        # pydub does things in milliseconds
        
        first = 0
        end = 30 * 1000
        time = 22.5 * 1000
        for i in range(int(len(sound)/time)+1):
            end = min(len(sound), end)
            segment = sound[first:end]

            add_idx = "_" + str(idx) + ".wav"
            name = path2 + '/'  + f.replace(".wav", add_idx)
        
            print(name, first, end)
            
            idx += 1
            first += 22.5 * 1000
            end += 22.5 * 1000

            segment.export(name, format="wav")
