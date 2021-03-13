# -*- coding: utf-8 -*-
"""
### Reqirements
- keras >= 2.2.0 or tensorflow >= 1.13
- segmenation-models==1.0.*
- albumentations==0.3.0

"""
"""# Loading dataset

Dataset structure:
 - **train** images + segmentation masks
 - **validation** images + segmentation masks
 - **test** images + segmentation masks
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm
import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask

from data_preprocess import visualize, denormalize, Dataloder, Dataset, round_clip_0_1, get_training_augmentation, get_validation_augmentation, get_preprocessing

"""# Segmentation Efficient-Unet model"""

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

class EffUnet():
    def __init__(self,
                 backbone='efficientnetb5',
                 batch_size=1,
                 classes=["houses", "buildings", "sheds"],
                 DATA_DIR="./incubit_data/",
                 ):
        self.BACKBONE = backbone
        self.BATCH_SIZE = batch_size
        self.CLASSES = classes

        self.x_train_dir = os.path.join(DATA_DIR, 'train')
        self.y_train_dir = os.path.join(DATA_DIR, 'trainannot')

        self.x_valid_dir = os.path.join(DATA_DIR, 'val')
        self.y_valid_dir = os.path.join(DATA_DIR, 'valannot')

        self.x_test_dir = os.path.join(DATA_DIR, 'test')
        self.y_test_dir = os.path.join(DATA_DIR, 'testannot')

        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)

# define network parameters
        self.n_classes = 1 if len(self.CLASSES) == 1 else (len(self.CLASSES) + 1)  # case for binary and multiclass segmentation
        self.activation = 'sigmoid' if self.n_classes == 1 else 'softmax'

#create model
        self.model = sm.Unet(self.BACKBONE, classes=self.n_classes, activation=self.activation, encoder_weights='imagenet', encoder_freeze=True)
        self.models = None
        self.history = None

        # model = sm.PSPNet(backbone_name=BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)
        self.valid_dataset = Dataset(
                                self.x_valid_dir, 
                                self.y_valid_dir, 
                                classes=self.CLASSES, 
                                augmentation=get_validation_augmentation(),
                                preprocessing=get_preprocessing(self.preprocess_input),
                                )

    def plot_train_stats(self,model_history):
        """
        Plot training and validation IOU scores and training Losses

        """
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(model_history.history['iou_score'])
        plt.plot(model_history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


    def train(self, weights_dir,
              model_type='single',
              INPUT_SIZE=768, n_models=7,
              learning_rate=0.0001,
              epochs=40,
              visualize_trainstats=False):
        optim = keras.optimizers.Adam(learning_rate)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([2, 1, 3, 0.5])) 
        focal_loss = sm.losses.BinaryFocalLoss() if self.n_classes == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        valid_dataloader = Dataloder(self.valid_dataset, batch_size=1, shuffle=False)


        # compile keras model with defined optimozer, loss and metrics
        if model_type == 'single':
            self.model.compile(optim, total_loss, metrics)

    # Dataset for train images
            INPUT_SIZE = 768
            train_dataset = Dataset(
                                    self.x_train_dir, 
                                    self.y_train_dir, 
                                    classes=self.CLASSES, 
                                    augmentation=get_training_augmentation(size=INPUT_SIZE),
                                    preprocessing=get_preprocessing(self.preprocess_input),
                                    )

    # Dataset for validation images

            train_dataloader = Dataloder(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

    # check shapes for errors
    # assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    # assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

            assert train_dataloader[0][0].shape == (self.BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 3)
            assert train_dataloader[0][1].shape == (self.BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, self.n_classes)


    # define callbacks for learning rate scheduling and best checkpoints saving
            callbacks = [
            keras.callbacks.ModelCheckpoint(weights_dir + 'unet_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
            keras.callbacks.ReduceLROnPlateau(),
            ]

    # train model
            history = self.model.fit_generator(
                                          train_dataloader, 
                                          steps_per_epoch=len(train_dataloader),
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=valid_dataloader,
                                          validation_steps=len(valid_dataloader),
                                          )
            if visualize_trainstats:
                self.plot_train_stats(model_history=history)

            return self.model, self.history
        elif model_type == 'ensemble':

            """# Ensembling Training

            Creating an emsemble of 7 models, feeding input images of different sizes. Final results will be computed based on a vote of sorts from all these models.
            """
            self.models=[0]*n_models
            self.history=[0]*n_models
            for j in range(n_models):
                self.models[j] =  self.model
                self.models[j].compile(optim, total_loss, metrics)
  
                train_dataset = Dataset(
                                        self.x_train_dir, 
                                        self.y_train_dir, 
                                        classes=self.CLASSES, 
                                        augmentation=get_training_augmentation(size=32*(10+2*j)),
                                        preprocessing=get_preprocessing(self.preprocess_input),
                                        )

                if j >= 5:
                    self.BATCH_SIZE = 1
                train_dataloader = Dataloder(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
                callbacks = [
                keras.callbacks.ModelCheckpoint(weights_dir+'best_model_Unet_ensemble_{}.h5'.format(j), 
                                                save_weights_only=True, save_best_only=True, mode='min'),
                keras.callbacks.ReduceLROnPlateau(),
                ]

                print("Training initialized: model #",j)
                history[j] = (self.models[j].fit_generator(
                                                      train_dataloader, 
                                                      steps_per_epoch=len(train_dataloader), 
                                                      epochs=epochs, 
                                                      callbacks=callbacks, 
                                                      validation_data=valid_dataloader, 
                                                      validation_steps=len(valid_dataloader),
                                                      ))
                if visualize_trainstats:
                    self.plot_train_stats(model_history=history[j])

            return self.models, self.history
        else:
            raise ValueError("Model type must be either 'single' or 'ensemble'. Please retry with appropriate model type")

    def run(self, output_dir, weights_dir,
            model_type='ensemble', n_models=7, 
            visualize_bool=False):

        output_masks = []
        file_names = []

        test_dataset = Dataset(
                               self.x_test_dir, 
                               self.y_test_dir, 
                               classes=self.CLASSES, 
                               augmentation=get_validation_augmentation(),
                               preprocessing=get_preprocessing(self.preprocess_input),
                               )

        test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

        for i in range(len(test_dataset)):
            image, gt_mask, file_name = test_dataset[i]
            file_names.append(file_name)
            image = np.expand_dims(image, axis=0)

            if model_type == 'ensemble':
                results = np.zeros((1, 1728, 1600, 4))
                self.models=[0]*n_models
                self.history=[0]*n_models
                for j in range(n_models):
                    self.models[j] = self.model
                    self.models[j].load_weights(weights_dir + 'best_model_Unet_ensemble_{}.h5'.format(j))
                    results = results + self.models[j].predict(image)
                pr_mask = results/n_models
            elif model_type=='single':
                self.model.load_weights(weights_dir + 'unet_model.h5')
                pr_mask = self.model.predict(image)
            
            else:
                raise ValueError("Model type must be either 'single' or 'ensemble'. Please retry with appropriate model type")

            output_masks.append(pr_mask)
            # segged_mask = self.segment_countours(image.squeeze(), pr_mask.squeeze())
            predicted_annotations, segged_mask = self.poly_from_mask(pr_mask, file_name, output_dir)

            
            if visualize_bool:
                visualize(
                          image=denormalize(image.squeeze()),
                          gt_mask=gt_mask.squeeze(),
                          pr_mask=pr_mask.squeeze(),
                          house_mask=pr_mask[...,0].squeeze(),
                          building=pr_mask[...,1].squeeze(),
                          shed_mask=pr_mask[...,2].squeeze(),
                          bg_mask=pr_mask[...,3].squeeze(),
                          segmented_mask = segged_mask,
                          )
        
        return output_masks, file_names

    def binarize_mask(self, mask_for_binarization, threshold=0.6, struct_type=0):
        mask_thresh = np.array(mask_for_binarization.squeeze())
        mask_thresh = np.where(mask_thresh>threshold,1, 0)
        # print(mask_thresh.shape)
        return mask_thresh


    def poly_from_mask(self, input_mask, file_name, output_dir):
        label_dict = {0:'Houses',1:'Buildings',2:'Sheds/Garages'}
        color_dict = {0:(0,0,255),1:(0,0,255),2:(255,0,0)}

        # ground_truth_binary_mask = binarize_mask(input_mask)
        allchannel_binary_mask = self.binarize_mask(input_mask)
        orig_image = cv2.imread(file_name)

        annotation = {
        "filename":file_name,
        "labels":[],
        # "segmentation": [],
        }
        for label in range(len(label_dict)):
            ground_truth_binary_mask = allchannel_binary_mask[...,label]
            # print(ground_truth_binary_mask[...,0].shape)
            fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask.astype(np.uint8))
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(ground_truth_binary_mask, 0.5)
            label_predictions = {"name":label_dict[label], "annotations":[]}
            # label_predictions = {"name":label_dict[label], "id":[], "segmentation":[]}
            for n,contour in enumerate(contours):
                contour = np.flip(contour, axis=1)
                cv2.drawContours(orig_image, [contour.astype(int)], -1, color_dict[label], thickness=5)
                segmentation = contour.ravel().tolist()
                # annotation["segmentation"].append(segmentation)
                # annotation["labels"]
                internal_dict = {"id":n,"segmentation":segmentation}
                label_predictions["annotations"].append(internal_dict)

            annotation["labels"].append(label_predictions)
        # plt.imshow(masker, cmap=plt.cm.gray)
        with open(output_dir + file_name.split('/')[-1].split(".")[0]+"_preds.json","w+") as fp:
            json.dump(annotation, fp)

        return annotation, orig_image
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EfficientNet based Unet model to segment satellite images.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'run'")
    parser.add_argument('--dataset', required=False,
                        default='./incubit_data/',
                        metavar="/path/to/satellite/dataset/",
                        help='Directory of the satellite dataset')
    parser.add_argument('--weights', required=False,
                        default='./incubit_data/weights/',
                        metavar="/path/to/weights/",
                        help="Directory/Path to (save or load weights) .h5 file or collection of h5 files (depending on single or ensemble)")
    parser.add_argument('--visualize', required=False,
                        default=False,
                        metavar="<visualize>",
                        help='To visualize training progress or inference masks')
    parser.add_argument('--model_type', required=False,
                        default='ensemble',
                        metavar="Specify type of model to train/run (ensemble or single)",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--output_dir', required=False,
                        default='./incubit_data/json_outs/',
                        metavar="/path/to/save_output/",
                        help='Directory to save json output')
    args = parser.parse_args()

    # DATA_DIR = './incubit_data/'
    model = EffUnet(DATA_DIR=args.dataset)
    print("EffUnet initialized successfully")

    if args.command == "train":
        model.train(model_type=args.model_type, weights_dir=args.weights, visualize_trainstats=args.visualize)
    else:
        model.run(output_dir=args.output_dir, weights_dir=args.weights, model_type=args.model_type, visualize_bool=args.visualize)