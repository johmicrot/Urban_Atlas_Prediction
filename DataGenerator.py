"""Creator: Daniel Pototzky,
   Authors: Daniel Pototzky, John Rothman"""
import keras
import numpy as np
from skimage.io import imread
from scipy.ndimage.interpolation import affine_transform


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, dataframe, directory, istrain, mode, class_name,  n_classes1=5, n_classes2=2,
                 batch_size=32, dim=(224, 224, 3), n_channels=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        self.n_classes2 = n_classes2
        self.directory = directory
        self.shuffle = shuffle
        self.istrain = istrain
        self.on_epoch_end()
        self.mode = mode
        self.class_name = class_name
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def rotate(self, img, row_axis=0, col_axis=1, channel_axis=2):
        theta = np.random.randint(0, 90)
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix
        h, w = img.shape[row_axis], img.shape[col_axis]
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, h, w)
        x = np.rollaxis(img, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [affine_transform(x_channel,
                                           final_affine_matrix,
                                           final_offset,
                                           order=1) for x_channel in x
                          ]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return img

    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def augment(self, img, istrain):
        # Numbers calculated form mean and std of Urban Atlas Dataset
        img[:, :, 0] = (img[:, :, 0] - 56.76211554) / 22.71095346
        img[:, :, 1] = (img[:, :, 1] - 65.80346362) / 24.25872165
        img[:, :, 2] = (img[:, :, 2] - 56.60189766) / 24.4900414

        if istrain:
            if np.random.random() < 0.5:
                img = self.flip_axis(img, 0)
            elif np.random.random() < 0.5:
                img = self.flip_axis(img, 1)

        return img

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_input_and_output(self, idx):
        df2 = self.dataframe
        img = imread(self.directory + '/' + df2.loc[idx, 'filename'])
        img = self.augment(img, self.istrain)
        if self.mode == 'single_task':
            label1 = df2.loc[idx, self.class_name]
            return img, label1
        elif self.mode == 'multi_task':
            label1 = df2.loc[idx, self.class_name[0]]
            label2 = df2.loc[idx, self.class_name[1]]
            # label2 = df2.loc[idx, self.class_name[1]]
            return img, label1, label2

        # return img, label1, label2

    def __data_generation(self, tmp_indices):
        mode = self.mode
        batch_x = []
        if mode == 'single_task':
            batch_y = []
            # Generate data
            for ID in tmp_indices:
                # Store sample
                input, output = self.get_input_and_output(ID)
                batch_x += [input]
                batch_y += [output]
            batch_x = np.array(batch_x)
            batch_y = np.asarray(batch_y)
            return batch_x, batch_y

        if mode == 'multi_task':
            batch_y1 = []
            batch_y2 = []
            # Generate data
            for ID in tmp_indices:
                # Store sample
                input, output1,  output2 = self.get_input_and_output(ID)
                batch_x += [input]
                batch_y1 += [output1]
                batch_y2 += [output2]
            batch_x = np.array(batch_x)
            batch_y1 = np.asarray(batch_y1)
            batch_y2 = np.asarray(batch_y2)
            return batch_x, [batch_y1, batch_y2]