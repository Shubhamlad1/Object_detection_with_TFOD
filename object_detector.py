import cv2 
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from pathlib import Path
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
#from object_detection.builders import model_builder
#from object_detection.utils import config_util
from loading_model import detect_fn
from paths import files
import base64



def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

class detector():
    def __init__(self, filepath):
        self.filepath = filepath

    def image_detection(self):
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        #IMAGE_PATH= os.path.join('Tensorflow', 'workspace','images')
        #img_path= os.path.join(IMAGE_PATH, 'train', 'WithCUP.7ab94cce-83b6-11ed-8819-98fa9bfba7d0.jpg')
        
        img = cv2.imread(self.filepath)
        

        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

        Img_rgb= cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        cv2.imwrite('color_img.jpg', Img_rgb)
        opencodedbase64 = encodeImageIntoBase64("color_img.jpg")
		# imagekeeper.append({"image": opencodedbase64.decode('utf-8')})

        result= {"image" : opencodedbase64.decode('utf-8')}

        return result