# Deep Learning: Binary Image Classification using CNN

## Problem Statement

Image classification is one of the most widely studied problems in computer vision. One common benchmark problem is distinguishing between images of cats and dogs, which requires the model to learn complex visual patterns such as shapes, textures, and edges.

The goal of this project is to develop a **Convolutional Neural Network (CNN)** to perform binary image classification using the **Dogs vs Cats dataset**. The dataset contains thousands of labeled images of cats and dogs with varying backgrounds, lighting conditions, and poses.

The CNN model automatically learns relevant visual features from raw images and classifies each image into one of two categories: **cat or dog**.

The project includes dataset preprocessing, CNN model design, training and validation, and evaluation using metrics such as **accuracy, precision, recall, and F1 score**.

**Expected Outcome:** A trained CNN model capable of accurately distinguishing between cats and dogs, typically achieving 80–95% accuracy depending on architecture and preprocessing techniques.

---

## Dataset Description

Each image in the dataset belongs to one of two classes and is loaded via `ImageFolder`, which reads class labels directly from the subfolder structure.
The dataset has been put in gitignore

| Attribute | Description |
| :--- | :--- |
| **Classes** | `cat`, `dog` (2 classes, binary classification) |
| **Training set** | Images organized under `training_set/cats/` and `training_set/dogs/` |
| **Test set** | Images organized under `test_set/cats/` and `test_set/dogs/` |
| **Image format** | JPEG images, varying original dimensions and backgrounds |
| **Preprocessing** | Resized to `128 x 128`, converted to tensor, normalised to `(-1, 1)` per RGB channel |

**Dataset Access**
The dataset for this project is excluded from version control via `.gitignore`. To run the code locally:

1. Download the dataset from https://www.kaggle.com/datasets/tongpython/cat-and-dog
2. Place the images so the folder structure reads `dataset_cat_dog/training_set/` and `dataset_cat_dog/test_set/`, each containing `cats/` and `dogs/` subfolders, directly in the root directory of this repository.
3. Execute the Jupyter Notebook.
