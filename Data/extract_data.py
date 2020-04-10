import numpy as np
import os


def file_to_numpy(path_data, path_label, labels):
    oslistval = os.listdir(path_data)
    list_classes = []
    for classe in oslistval:
        path_data_classe = path_data + "/" + classe
        path_label_classe = path_label + "/" + classe
        oslistfile = os.listdir(path_data_classe)
        list_array_classe = []
        for file in oslistfile:
            # open file .pts (data) and store it
            path_file = path_data_classe + "/" + file
            file_array = np.fromfile(path_file, dtype=float, count=-1, sep=' ', offset=0)
            file_array = file_array.reshape((-1, 3))
            # add object label and the label of the part of the object
            object_label = np.ones((file_array.shape[0], 1)) * labels.index(classe)
            part_label = np.zeros((file_array.shape[0], 1))  # will be completed after
            cols = np.hstack((object_label, part_label))
            file_array = np.hstack((file_array, cols))
            # complete the label of the part of the object
            path_seg_file = path_label_classe + "/" + file.split("pts")[0] + "seg"
            with open(path_seg_file, 'r') as f:
                i = 0
                for line in f:
                    file_array[i, 4] = int(line)
                    i += 1
            list_array_classe.append(file_array)
        # store data from all files in one array
        list_array_classe = np.vstack(list_array_classe)
        list_classes.append(list_array_classe)
    # store data from all classes in one array
    array_classes = np.vstack(list_classes)
    return array_classes


# labels of the sixteen classes
labels = ["02691156",	"02773838", "02954340", "02958343",	"03001627", "03261776",	"03467517", "03624134",
          "03636649",	"03642806",	"03790512",	"03797390", "03948459", "04099429", "04225987", "04379243"]


path_val_data = "données/val_data"
path_val_label = "données/val_label"
path_train_data = "données/train_data"
path_train_label = "données/train_label"

# get validation data
array_classes = file_to_numpy(path_val_data, path_val_label, labels)
np.save("données/val_data_classe3.npy", array_classes[:, 0:3])
np.save("données/val_label_classe3.npy", array_classes[:, 4])

# get train data
array_classes = file_to_numpy(path_train_data, path_train_label, labels)
np.save("données/train_data_classe3.npy", array_classes[:, 0:3])
np.save("données/train_label_classe3.npy", array_classes[:, 4])
