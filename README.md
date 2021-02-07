# You Only Look Once (YOLO) v3 

### Introduction

YOLO is a state-of-the-art, real-time object detection algorithm. In this notebook, we apply the YOLO algorithm to detect objects in images. Test images are located in the`./images/`folder.

### Importing Resources

We will start by loading the required packages into Python. We will be using *OpenCV* to load our images, *matplotlib* to plot them, a`utils` module that contains some helper functions, and a modified version of *Darknet*. YOLO uses *Darknet*, an open source, deep neural network framework written by the creators of YOLO. The version of *Darknet* used in this notebook has been modified to work in PyTorch 0.4 and has been simplified because we won't be doing any training. Instead, we will be using a set of pre-trained weights that were trained on the Common Objects in Context (COCO) database. For more information on *Darknet*, please visit <a href="https://pjreddie.com/darknet/">Darknet</a>.

### Setting Up The Neural Network

We will be using the latest version of YOLO, known as YOLOv3. The `yolov3.cfg` file contains the network architecture used by YOLOv3 in the `/cfg/` folder. Similarly, the `yolov3.weights` file contains the pre-trained weights in the `/weights/` directory. Finally, the `/data/` directory, contains the `coco.names` file that has the list of the 80 object classes that the weights were trained to detect.

In the code below, we start by specifying the location of the files that contain the neural network architecture, the pre-trained weights, and the object classes.  We then use *Darknet* to setup the neural network using the network architecture specified in the `cfg_file`. We then use the`.load_weights()` method to load our set of pre-trained weights into the model. Finally, we use the `load_class_names()` function, from the `utils` module, to load the 80 object classes.

### `.print_network()` 
The neural network used by YOLOv3 consists mainly of convolutional layers, with some shortcut connections and upsample layers. For a full description of this network please refer to the <a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf">YOLOv3 Paper</a>.

### Loading and Resizing Images
we load our images using OpenCV's `cv2.imread()` function. Since, this function loads images as BGR we will convert our images to RGB so we can display them with the correct colors.

The input size of the first layer of the network is 416 x 416 x 3. Since images have different sizes, we have to resize our images to be compatible with the input size of the first layer in the network. In the code below, we resize our images using OpenCV's `cv2.resize()` function. 

### Non-Maximal Suppression Threshold

YOLO uses **Non-Maximal Suppression (NMS)** to only keep the best bounding box. The first step in NMS is to remove all the predicted bounding boxes that have a detection probability that is less than a given NMS threshold. For example, when NMS threshold is set to `0.6`: This means that all predicted bounding boxes that have a detection probability less than 0.6 will be removed. 

###  IOU(Intersection Over Union)Threshold
After removing all the predicted bounding boxes that have a low detection probability, the second step in NMS, is to select the bounding boxes with the highest detection probability and eliminate all the bounding boxes whose **Intersection Over Union (IOU)** value is higher than a given IOU threshold. for examples, if IOU threshold is set to `0.4`: this means that all predicted bounding boxes that have an IOU value greater than 0.4 with respect to the best bounding boxes will be removed.
In the `utils` module the `nms` function performs the second step of Non-Maximal Suppression, and the `boxes_iou` function calculates the Intersection over Union of two given bounding boxes.

### Object Detection

Once the image has been loaded and resized, and parameters for `nms_thresh` and `iou_thresh` need to be set to use YOLO algorithm for object detection in the image. We detect the objects using the `detect_objects(m, resized_image, iou_thresh, nms_thresh)`function from the `utils` module. This function takes in the model `m` returned by *Darknet*, the resized image, and the NMS and IOU thresholds, and returns the bounding boxes of the objects found.

Each bounding box contains 7 parameters: the coordinates *(x, y)* of the center of the bounding box, the width *w* and height *h* of the bounding box, the confidence detection level, the object class probability, and the object class id. The `detect_objects()` function also prints out the time it took for the YOLO algorithm to detect the objects in the image and the number of objects detected. Since we are running the algorithm on a CPU it takes about 2 seconds to detect the objects in an image, however, if we were to use a GPU it would run much faster.

Once we have the bounding boxes of the objects found by YOLO, we can print the class of the objects found and their corresponding object class probability. To do this we use the `print_objects()` function in the `utils` module.

Finally, we use the `plot_boxes()` function to plot the bounding boxes and corresponding object class labels found by YOLO in our image. If the `plot_labels` flag is set to `False` the display the bounding boxes is with no labels. This makes it easier to view the bounding boxes if `nms_thresh` is too low. The `plot_boxes()`function uses the same color to plot the bounding boxes of the same object class. However, to make all bounding boxes to be the same color, the `color` keyword can be set to the desired color. 

### result 
![input image](/images/img2.png)
![result image](/images/result.png)

