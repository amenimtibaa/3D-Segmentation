"""
This file is to process the shapenet dataset to return train and validation set
the label of each set vary between 1 and 50 ( 50 is the total number of subcategories)
"""
import numpy as np
import os
np.set_printoptions(suppress=True)

def compute_offset(category):
    """
    Return a number between 1 and 50
    """
    # get index
    index = labels.index(category)
    # get all previous categories
    previous = labels[:index]
    # get total nbr of part for all the previous categories
    total = 0
    for cat in previous:
        total += Nbr_part_per_category[cat]
    return total

def select_points(points, num_points):
    """
    :param points: point cloud
    :param num_points: maximum number of points to select from point cloud
    :return:
    """
    # shuffle points and generate index
    if points.shape[0] > num_points:
        index = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        index = np.random.choice(points.shape[0], num_points, replace=True)
    selected_points = np.take(points, index, axis=0)
    return selected_points


def file_to_numpy(path_data, path_label, nb_pts_in_cloud=1024):

    oslistval = os.listdir(path_data)
    list_classes = []

    # parse categories
    for classe in oslistval:

        path_data_classe = path_data + "/" + classe
        path_label_classe = path_label + "/" + classe
        # oslistfile contains (.pts) files
        oslistfile = os.listdir(path_data_classe)
        list_array_classe = []
        # Parse data files :
        for file in oslistfile:
            # open file
            path_file = path_data_classe + "/" + file
            file_array = np.fromfile(path_file, dtype=float, count=-1, sep=' ', offset=0)
            file_array = file_array.reshape((-1, 3))

            # add the label of the part of the object
            part_label = np.zeros((file_array.shape[0], 1))
            file_array = np.hstack((file_array, part_label))
            # complete the label of the part of the object
            path_seg_file = path_label_classe + "/" + file.split("pts")[0] + "seg"
            #compute the offset to add for each category
            category_offset = compute_offset(classe)

            with open(path_seg_file, 'r') as f:
                for i, line in enumerate(f):
                    # the second term of the addition allows the creation of unique labels
                    a = int(line) + category_offset
                    file_array[i, 3] = int(a)
                    #i += 1

            # shuffle the array and take nb_pts_in_cloud lines
            file_array = select_points(file_array, nb_pts_in_cloud)

            list_array_classe.append(np.reshape(file_array, (1, nb_pts_in_cloud, -1)))

        # store data from all files in one array
        list_array_classe = np.vstack(list_array_classe)
        list_classes.append(list_array_classe)

    # store data from all classes in one array
    array_classes = np.vstack(list_classes)

    return array_classes


# labels of the sixteen classes (categories)
labels = ["02691156",	"02773838", "02954340", "02958343",	"03001627", "03261776",	"03467517", "03624134",
          "03636649",	"03642806",	"03790512",	"03797390", "03948459", "04099429", "04225987", "04379243"]

Nbr_part_per_category = {'04225987': 3, '02958343': 4, '03790512': 6, '03797390': 2,
 '03624134': 2, '03636649': 4, '02691156': 4, '02954340': 2, '03948459': 3,
 '03642806': 2, '04379243': 3, '03001627': 4, '03261776': 3, '04099429': 3,
 '02773838': 2, '03467517': 3}

"""
category_to_label = {'Cap': '02954340', 'Rocket': '04099429', 'Lamp': '03636649', 'Motorbike': '03790512',
                     'Car': '02958343', 'Airplane': '02691156', 'Skateboard': '04225987', 'Mug': '03797390',
                     'Laptop': '03642806', 'Bag': '02773838', 'Guitar': '03467517', 'Earphone': '03261776',
                     'Pistol': '03948459', 'Knife': '03624134', 'Table': '04379243','Chair': '03001627'}
"""

nb_pts_per_cloud = 2048

path_val_data = "data/val_data"
path_val_label = "data/val_label"
path_train_data = "data/points"
path_train_label = "data/points_label"

print(' ***************************** ')
# get train data
array_classes = file_to_numpy(path_train_data, path_train_label, nb_pts_per_cloud)
np.save("train_data.npy", array_classes[:,:, 0:3])
np.save("train_labels.npy", array_classes[:, :,3])
print("train data : ", array_classes[:,:, 0:3].shape)


# get validation data
array_classes = file_to_numpy(path_val_data, path_val_label, nb_pts_per_cloud)
np.save("val_data.npy", array_classes[:,:, 0:3])
np.save("val_label.npy", array_classes[:, :,3])
print("val data : ", array_classes[:,:, 0:3].shape)


