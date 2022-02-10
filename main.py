import numpy as np
import pathlib
import os
from keras.preprocessing.image import ImageDataGenerator

NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE = 60
NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE_TRAIN = int(NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE * 0.8)
NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE_TEST = int(NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE * 0.2)

COINS_DATASET_TRAIN_PATH = "src/train/"
COINS_DATASET_TRAIN_PATH_AUGMENTED = "classified/train/"
COINS_DATASET_TEST_PATH_AUGMENTED = "classified/test/"

data_dir = pathlib.Path(COINS_DATASET_TRAIN_PATH)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

for class_name in class_names:
    augmented_class_train_data_path = COINS_DATASET_TRAIN_PATH_AUGMENTED + class_name
    augmented_class_test_data_path = COINS_DATASET_TEST_PATH_AUGMENTED + class_name

    os.mkdir(augmented_class_train_data_path)
    os.mkdir(augmented_class_test_data_path)

    datagen = ImageDataGenerator(
        rotation_range=180,
        brightness_range=[0.4, 1.5],
        fill_mode='nearest'
    )

    print("Creating " + str(NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE_TRAIN) +
          " training images for " + class_name + " class")

    i = 1
    for batch in datagen.flow_from_directory(COINS_DATASET_TRAIN_PATH,
                                             batch_size=32,
                                             target_size=(256, 256),
                                             save_to_dir=augmented_class_train_data_path,
                                             save_format='jpg',
                                             classes=[class_name]):
        i += 1
        if i > NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE_TRAIN:
            break

    print("Creating " + str(NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE_TEST) +
          " testing images for " + class_name + " class")

    i = 1
    for batch in datagen.flow_from_directory(COINS_DATASET_TRAIN_PATH,
                                             batch_size=32,
                                             target_size=(256, 256),
                                             save_to_dir=augmented_class_test_data_path,
                                             save_format='jpg',
                                             classes=[class_name]):
        i += 1
        if i > NUMBER_OF_IMAGES_PER_CLASS_AVAILABLE_TEST:
            break


