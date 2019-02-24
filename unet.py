import numpy as np

from keras.layers import Conv2D, BatchNormalization, MaxPool2D
from keras.layers import Input, concatenate, Conv2DTranspose
from keras.models import Model, save_model, load_model

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import CSVLogger

from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import cv2
from helper import clear_dir_contents
from helper import get_count


class Unet(object):
    """docstring for Unet"""

    def __init__(self, sz=64, depth=5):

        K.clear_session()

        self.img_rows, self.img_cols = sz, sz
        self.depth_of_unet = depth
        self.gpu = 1
        self.batch_size = 3 * self.gpu
        self.seed = 42

        self.train_img_path = 'data/train/images/'
        self.train_mask_path = 'data/train/masks/'
        self.validation_img_path = 'data/valid/images/'
        self.validation_mask_path = 'data/valid/masks/'
        self.test_img_path = 'data/test'

        # 4320,960
        self.train_samples = 250
        self.validation_samples = 50
        self.threshold = 127

        self.name = '2D_Unet'
        self.save_path = ''.join(['models/', self.name, '_best', '.h5'])
        self.log_path = ''.join(['logs/', self.name, '_log', '.csv'])

    def get_model(self):

        inputs = Input((self.img_rows, self.img_cols, 1))
        depth = self.depth_of_unet

        curr_layer = inputs
        layers = []
        f = 32

        for i in range(depth):

            conv1 = Conv2D(f, (3, 3), activation='relu',
                           padding='same')(curr_layer)
            bn1 = BatchNormalization()(conv1)
            curr_layer = bn1

            conv2 = Conv2D(f, (3, 3), activation='relu',
                           padding='same')(curr_layer)
            bn2 = BatchNormalization()(conv2)
            curr_layer = bn2

            if i < depth - 1:
                maxpool1 = MaxPool2D(pool_size=(2, 2))(curr_layer)
                curr_layer = maxpool1
                layers.append([bn1, bn2, maxpool1, f])
            else:
                layers.append([bn1, bn2, f])
            f *= 2

        for i in range(depth - 2, -1, -1):
            f = layers[i][-1]

            prev_conv = layers[i][1]
            upconv = concatenate([Conv2DTranspose(f, (2, 2), strides=(
                2, 2), padding='same')(curr_layer), prev_conv], axis=3)
            curr_layer = upconv

            conv1 = Conv2D(f, (3, 3), activation='relu',
                           padding='same')(curr_layer)
            bn1 = BatchNormalization()(conv1)
            curr_layer = bn1

            conv2 = Conv2D(f, (3, 3), activation='relu',
                           padding='same')(curr_layer)
            bn2 = BatchNormalization()(conv2)
            curr_layer = bn2

        del layers

        conv_output = Conv2D(1, (1, 1), activation='sigmoid')(curr_layer)
        model = Model(inputs=[inputs], outputs=[conv_output])
        model.summary()
        return model

    def get_dice_coeff(self, y_pred, y_true):
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)

        intersect = K.sum(y_pred * y_true)
        denominator = K.sum(y_pred) + K.sum(y_true)

        dice_coeff = (2.0 * intersect) / (denominator + 1e-6)
        return dice_coeff

    def custom_loss(self, y_pred, y_true):
        dice_coeff = self.get_dice_coeff(y_pred, y_true)
        return 1.0 - dice_coeff

    def build_model(self, lr=1e-4):
        opt = Adam(lr=lr)
        model = self.get_model()
        model.compile(
            optimizer=opt,
            # loss = 'binary_crossentropy',
            loss=self.custom_loss,
            metrics=['accuracy']
        )
        return model

    def load_model(self, path):
        model = load_model(path, custom_objects={
                           'custom_loss': self.custom_loss})
        # model = load_model(path)
        return model

    def train_gen(self, path):
        img_gen = ImageDataGenerator(
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            rotation_range=5.0,
            rescale=1 / 255.0)

        img_gen = img_gen.flow_from_directory(
            path,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode=None,
            color_mode='grayscale'
        )

        return img_gen

    def test_gen(self, path):
        img_gen = ImageDataGenerator(
            rescale=1 / 255.0)

        img_gen = img_gen.flow_from_directory(
            path,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode=None,
            color_mode='grayscale',
            shuffle=False,
        )

        return img_gen

    def get_train_generator(self):
        img_gen = self.train_gen(self.train_img_path)
        mask_gen = self.train_gen(self.train_mask_path)

        while True:
            images = next(img_gen)
            masks = next(mask_gen)
            masks = self.normalize_array(masks)
            masks = masks > self.threshold
            yield images, masks

    def get_validation_generator(self):
        img_gen = self.test_gen(self.validation_img_path)
        mask_gen = self.test_gen(self.validation_mask_path)
        while True:
            images = next(img_gen)
            masks = next(mask_gen)
            masks = self.normalize_array(masks)
            masks = masks > self.threshold
            yield images, masks

    def get_test_generator(self, path):
        img_gen = self.test_gen(path)
        return img_gen

    def get_callbacks(self, num_epochs):
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=1e-6, patience=10, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(
            filepath=self.save_path, verbose=1, save_best_only=True)
        # tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=4)
        csv_logger = CSVLogger(self.log_path)
        return [checkpointer, reduce_lr, early_stopping,csv_logger]

    def train(self, lr=1e-4, num_epochs=1):

        model = self.build_model(lr)

        train_generator = self.get_train_generator()
        validation_generator = self.get_validation_generator()

        hist = model.fit_generator(
            generator=train_generator,
            epochs=num_epochs,
            validation_data=validation_generator,
            callbacks=self.get_callbacks(num_epochs),
            verbose=1,
            steps_per_epoch=self.train_samples // self.batch_size + 1,
            validation_steps=self.validation_samples // self.batch_size + 1,
            pickle_safe=False
        )

    def continue_training(self, lr=1e-4, num_epochs=1):

        model = self.load_model(self.save_path)
        K.set_value(model.optimizer.lr, lr)

        train_generator = self.get_train_generator()
        validation_generator = self.get_validation_generator()

        hist = model.fit_generator(
            generator=train_generator,
            epochs=num_epochs,
            validation_data=validation_generator,
            callbacks=self.get_callbacks(num_epochs),
            verbose=1,
            steps_per_epoch=self.train_samples // self.batch_size + 1,
            validation_steps=self.validation_samples // self.batch_size + 1
        )

    def normalize_array(self, arr, lower=0, upper=255):
        diff = np.max(arr) - np.min(arr)
        if diff != 0:
            arr = (upper - lower) * (arr - np.min(arr)) / \
                diff.astype(np.float32)
        return arr.astype(np.float32)

    def get_predictions(self):
        model = self.load_model(self.save_path)
        validation_generator = self.get_validation_generator()

        steps = self.validation_samples // self.batch_size + 1

        image_array = np.empty(shape=(0, self.img_rows * self.img_cols))
        y_true = np.empty(shape=(0, self.img_rows * self.img_cols))
        y_pred = np.empty(shape=(0, self.img_rows * self.img_cols))

        for i in range(steps):
            print(steps - i)

            images, masks = next(validation_generator)
            predictions = model.predict(
                x=images,
                batch_size=self.batch_size
            )

            y_true = np.vstack(
                [y_true, masks.reshape(-1, self.img_rows * self.img_cols)])
            image_array = np.vstack(
                [image_array, images.reshape(-1, self.img_rows * self.img_cols)])
            y_pred = np.vstack(
                [y_pred, predictions.reshape(-1, self.img_rows * self.img_cols)])
            del predictions

        np.save('tmp/' + self.name + '_images.npy', image_array)
        np.save('tmp/' + self.name + '_ypred.npy', y_pred)
        np.save('tmp/' + self.name + '_ytrue.npy', y_true)

        y_pred = np.load('tmp/' + self.name + '_ypred.npy')
        y_true = np.load('tmp/' + self.name + '_ytrue.npy')
        image_array = np.load('tmp/' + self.name + '_images.npy')

        y_pred = self.normalize_array(y_pred, 0, 255)
        y_true = self.normalize_array(y_true, 0, 255)

        y_pred = y_pred > self.threshold
        y_true = y_true > self.threshold

        return y_true, y_pred, image_array

    def evaluate(self, output_folder='tmp/outputs/', visualize=True):

        clear_dir_contents(output_folder)
        y_true, y_pred, image_array = self.get_predictions()

        y_true = y_true.reshape(-1, self.img_rows, self.img_cols)
        y_pred = y_pred.reshape(-1, self.img_rows, self.img_cols)
        image_array = image_array.reshape(-1, self.img_rows, self.img_cols)

        if visualize:
            for idx in range(image_array.shape[0]):
                img = image_array[idx, :, :]
                true_mask = y_true[idx, :, :]
                gen_mask = y_pred[idx, :, :]

                out_img = np.hstack(
                    [true_mask * 255, img * 255, gen_mask * 255, (true_mask ^ gen_mask) * 255])
                name = output_folder + str(idx) + '.jpg'
                print(name)
                cv2.imwrite(name, out_img)

        y_pred = K.variable(y_pred)
        y_true = K.variable(y_true)
        loss = self.custom_loss(y_pred, y_true)
        dice_coeff = 1 - K.eval(loss)
        print('Dice Coeff : ', dice_coeff)

    def debug(self):

        clear_dir_contents('tmp/debug/')
        # gen = self.get_train_generator()
        gen = self.get_validation_generator()

        steps = self.train_samples // self.batch_size
        for i in range(steps)[:10]:
            images, masks = next(gen)

            for idx in range(len(images)):
                img = images[idx].reshape(
                    self.img_rows, self.img_cols).astype(np.float32)
                mask = masks[idx].reshape(
                    self.img_rows, self.img_cols).astype(np.float32)

                img = self.normalize_array(img, 0, 255)
                mask = mask * 255

                print(img.shape)
                print(np.unique(mask))

                name = 'tmp/debug/' + str(i) + '_' + str(idx) + '_img.jpg'
                cv2.imwrite(name, img)

                name = 'tmp/debug/' + str(i) + '_' + str(idx) + '_mask.jpg'
                cv2.imwrite(name, mask)


if __name__ == '__main__':

    u1 = Unet(sz=1024)
    # u1.debug()
    u1.train(lr=1e-4, num_epochs=50)
    # u1.continue_training(lr=1e-6, num_epochs=2)
    u1.evaluate(visualize=True, output_folder='tmp/normal_2d/')

    # u1.evaluate(visualize=False)
    # u1.continue_training(lr=1e-4,num_epochs=100)
    # u1.evaluate(visualize=False)
    # u1.continue_training(lr=1e-5,num_epochs=20)
    # get_count('tmp/')
