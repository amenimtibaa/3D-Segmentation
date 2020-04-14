# 3D Part Segmentation

Keras implementation of pointnet for part segmentation on shapenet dataset.

## Getting Started

These instructions will get you a copy of the project up and running for development and testing purposes. 

### Download data

Firstly you need to download the data from [here](https://shapenet.cs.stanford.edu/iccv17/) : (Training Point Clouds | Training Label | Validation Point Clouds | Validation Label).

Then unzip this folder and put them in a directory called "data" under the ShapeNet directory.

### Prepare train & validation dataset

Shapenet contains 16 shape categories. Each category is annotated with 2 to 6 parts and there are 50 different parts annotated in total.
If you open the train_labels data directory you will find 16 sub-directories where each one contains ".seg" files and each line of this files contains a number betwenn 0 and 6. what we want is to associate a unique number to each part before creating the dataset.
This is done in "process_data.py", after executing it you should have 4 "npy" arrays added ( train data/labels and validation data/labels). 

### Running the tests

Now, you only need to upload the data to your drive and run the colab notebook or save it as jupyter notebook and execute it localy :)

## some results

 * good results
 
  ![](https://drive.google.com/uc?id=1LDoK-7pn7mfp43EF4ShK_u1YHWGpxATZ)
  
  ![](https://drive.google.com/uc?id=1r7gttzbzW87NF9Bg2-CFSefjnVxfUS_4)
  
  ![](https://drive.google.com/uc?id=1yx0QzC5KQRptajJP18gPF35u40123cpn)
  
 * meduim/bad results
 
  ![](https://drive.google.com/open?id=1z8raJI0wQ142RNkrx7yIVfEC6jJ91WWB)
  
  ![](https://drive.google.com/open?id=1aP-TI-M6I6JCQdgmanc6nrlJoBWaW986)


## Acknowledgments

* Original tensorflow code is [here](https://github.com/charlesq34/pointnet)

* This project is part of the [3D course](http://caor-mines-paristech.fr/fr/cours-npm3d/) of the [IASD master](https://www.lamsade.dauphine.fr/wp/iasd/en/).

