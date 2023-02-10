import os
import json
import pickle

## Create DOS data
folder_path_dir = "./raw/dos_ft"
file_list = os.listdir(folder_path_dir)
print(len(file_list))

dosdata = {}
for file in file_list:
    try :
        with open("./dosdata/{}".format(file), "r") as f:
            json_data = json.load(f)
            dosdata[file[:-9]] = json_data
    except :
        pass

with open("dos.pkl","wb") as f:
    pickle.dump(dosdata, f)

print("DOS data created! --> total {} | success : {} | fail : {}".format(len(file_list), len(dosdata), len(file_list) - len(dosdata)))


## Create MP data
folder_path_dir = "./raw/mpdata"
file_list = os.listdir(folder_path_dir)
print(len(file_list))

mpdata = {}
for file in file_list:
    try :
        with open("./mpdata/{}".format(file), "r") as f:
            json_data = json.load(f)
            mpdata[json_data["response"][0]["material_id"]] = json_data["response"][0]
    except :
        pass

with open("mp.pkl","wb") as f:
    pickle.dump(mpdata, f)

print("MP data created! --> total {} | success : {} | fail : {}".format(len(file_list), len(mpdata), len(file_list) - len(mpdata)))