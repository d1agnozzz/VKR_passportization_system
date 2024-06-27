import os

import numpy as np
import tensorflow as tf

import panoptic_visualization
import cv2


class PanopticDeeplabInference():
    def __init__(self, model_dir: str):
        self.model = tf.saved_model.load(model_dir)

    def evaluateSegmentation(self, rgb_img, slam_masking: bool = True) -> np.ndarray:
        input_tensor = tf.convert_to_tensor(rgb_img)
        
        result_dict = self.model(input_tensor)
        masked_img, _ = panoptic_visualization.color_panoptic_map(tf.squeeze(result_dict['panoptic_pred'],0), panoptic_visualization.cityscapes_dataset_information(), perturb_noise=60, slam_masking=slam_masking)
        
        return np.bitwise_and(rgb_img, masked_img)
        


def test_deeplab_with_panoptic_deeplab():

    # tf.keras.backend.set_floatx('float16')
    # model, experiment_options = _create_model_from_test_proto(
        # 'resnet50_os32_merge_with_pure_tf_func.textproto')
    model = tf.saved_model.load('/home/ubuntuser/panoptic/swidernet_sac_1_1_4.5_os16_panoptic_deeplab_cityscapes_trainfine_saved_model')
    # model = tf.train.load_checkpoint()

    rgb_image = cv2.resize(cv2.cvtColor(cv2.imread("/media/ubuntuser/bf3469a2-e299-475d-b7fe-3ccd74cf24de/klepolin/Haruna screenshots/track_1p1-0001.jpg"), cv2.COLOR_BGR2RGB), (1920, 1080), interpolation=cv2.INTER_AREA)#.astype(np.float32)
    
    batch = np.expand_dims(rgb_image, 0)

    input_tensor = tf.convert_to_tensor(rgb_image)
    # input_tensor = tf.random.uniform(
        # shape=(2, train_crop_size[0], train_crop_size[1], 3))
    resulting_dict = model(input_tensor)

    print(resulting_dict['panoptic_pred'])

    panoptic_visualization.vis_segmentation(rgb_image, tf.squeeze(resulting_dict['panoptic_pred'], 0), dataset_info=panoptic_visualization.cityscapes_dataset_information(), perturb_noise=0, masking_slam=True)


if __name__ == '__main__':
#   tf.test.main()
    test_deeplab_with_panoptic_deeplab()