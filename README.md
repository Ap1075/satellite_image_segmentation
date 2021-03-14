<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#data">Data</a></li>
         <li><a href="#labels">Labels</a></li>
         <li><a href="#proposed-solution">Proposed Solution</a></li>    
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!-- <li><a href="#roadmap">Roadmap</a></li> -->
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <!-- <li><a href="#license">License</a></li> -->
    <!-- <li><a href="#contact">Contact</a></li> -->
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

You are given a small slice of satellite data of a city in Japan by a fantasy land surveying company. They want to evaluate the feasibility of using satellite data to augment some downstream tasks. To that end, they need to segment out individual buildings, by category. They also need to return the segmentation in vector format so that they can import it into their CAD software.

They had some of their interns annotate some of the data and would like you to have a go at it. They're not data scientists so the data they provided might not be optimal and their annotations not entirely consistent. But, it is what it is, and you have to make due with it.


### Data:
The data consists of one single sattelite picture of Tokyo split into a 9x9 grid (81 non-overlapping PNG images total). The naming convention reflects it. You can put it back together if you want to.

In addition, you receive 72 annotation data files containing target labels for individual images. 9 of them, forming the bottom right corner of the image, are kept away for evaluation.

The data is picked in a way that will alleviate model training time, while still being consistent enough to have a good chance of yielding reasonable output on the test data.

### Labels:
The annotated data consists of three labels that are of interest in the image:
1. Houses
2. Buildings
3. Sheds/Garages

The labels come from a custom annotation tool. The format doesn't follow any standard academic data format, but it's pretty straightforward.

* The label data is provided in json format. One json annotation file per image, named after the image it represents. 

* Each file is a dictionary of lists wherein the lists are in the order: Houses, Buildings and Sheds/Garages. Each construction structure has a corresponding dictionary containing annotations.

* The labels are provided as polygons under an [x,y,x,y,x,y....] format. Once the sequence is finished, the last point connects to the first point. There is no distinction between clockwise and conter-clockwise.

### Proposed Solution:
The problem is solved as a multi-class segmentation problem using:
* a U-Net architecture with an EfficientNet encoder ([imagenet](http://image-net.org/) pretrained weights). Models were trained on significantly augmented data due to small dataset size.
* The final solution is an ensemble of 7 models, each trained on progressively larger input images and finally an average is computed to attain the segmentation prediction.
* Output in the form of json files with the file_name and labels. Masks can be visualized for inspection as well.
* Interactive demo/report available in this[ jupyter notebook](https://github.com/Ap1075/satellite_image_segmentation/blob/main/report.ipynb).

## Getting Started

To get a local copy up and running follow these simple steps:

### Prerequisites

The code was developed and tested on:
* Python 3.7.10
* Cuda 11.2
* Nvidia driver version: 460.32.03.
* OpenCV 4.1.2

### Installation

1. Clone the repo
```sh
git clone https://github.com/Ap1075/satellite_image_segmentation.git

```

2. Get required python packages (**Recommended**: Create and activate a virtualenv)

```sh
pip install -r requirements.txt
```

## Usage
1. Create annotation masks from json annotation files. These annotation masks will then be used to train the model. Masks will be placed in "annotation_masks" subdirectory.

```sh
python masks_from_json.py
```
2. Setup the required directory structure to ease usage. What's happening inside?
* Creates main directory "incubit_data" and subdirectories for train, validation, test data along with weights and output files. 
* Fills subdirectories (except weights) with the required images and annotation masks.

```sh
. setup.sh
```
Download weights from [drive](https://drive.google.com/drive/folders/1CWsW9CjVYDGz46VJ-zOW5V3dRSbEL93f?usp=sharing) and place in the 'weights' subdirectory.

3. You can either train your own model or run inference using existing models.
* To train your own (single or ensemble): (**NOTE**: all arguments have default values. For eg: weights stored in ./incubit_data/weights by default.)
```sh
python EffUnet.py train --model_type "ensemble" --visualize True --weights path/to/save_weights

```
* To run inference using a trained model (single or ensemble):
```sh
python EffUnet.py run --model_type "ensemble" --visualize True --weights path/to/load_weights

```

## Acknowledgements
1. [Segmentation Models](https://github.com/qubvel/segmentation_models)
2. [SpaceNet challenge solutions](https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions)
3. [Eff-Unet Paper, CVPR 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Baheti_Eff-UNet_A_Novel_Architecture_for_Semantic_Segmentation_in_Unstructured_Environment_CVPRW_2020_paper.pdf)