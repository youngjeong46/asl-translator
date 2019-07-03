# Sign Language Translator


## Overview

This project developed an application that can translate ASL alphabet into English alphabet and prints it on-screen.

**Blog post on the topic can be found [here](https://datatostories.com/posts/2019/06/25/asl-translator/).**

## Project Design

Using a self-made dataset of 100 images each of 29 gestures (26 English alphabets + "Space", "Backspace", "Nothing"), I trained two architectures: VGG16 and MobileNet V1. Instead of training all the weights, the pre-weights were loaded from the ImageNet dataset and Transfer Learning was applied.

## Application

Using OpenCV, I coded a script in Python that takes live video where a gesture inside a box is recognized and printed on screen for ASL translation into English language.

![script](https://datatostories.s3-us-west-2.amazonaws.com/asl-app-screenshot.png)

The accuracy for VGG16 was at 94% and MobileNet V1 was at 92%. However, MobileNet V1 was about 4-5 times faster than VGG16 and required less than 10% (32MB) of the memory that VGG16 required (454MB). For the application, I chose to go with MobileNet V1 as the architecture.

## Tools

- Data Storage: AWS S3
- Data Visualization: TensorBoard
- Deep Learning: Keras w/ Tensorflow Backend
- Application: Python, OpenCV
- Presentation: Google Slides

## Project Organization

The structure of the project organization is adapted from Cliff Clive's [DataScienceMVP](https://github.com/cliffclive/datasciencemvp), which itself is adapted from the famous Data Science project structure from Cookiecutters that you can access [here](https://github.com/drivendata/cookiecutter-data-science/).

```
    ├── README.md          <- The top-level README for developers using this project.
    ├── models             <- Keras model files saved in .h5 format
    │  
    ├── references         <- Trained and serialized models, model predictions, or model summaries
    │   ├── models         <- Generated model records stored in pickle files
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── Models.ipynb   <- General overview of the projectin Jupyter notebook.
    │
    ├── reports            <- Various reports
    │   ├── demo.avi       <- demo video of the python script
    │   └── proposal.pdf   <- proposal of the project.
    ├── src                <- Various source codes
    │   ├── demo.py        <- script that translates gestures into English (for demo video purposes)
    │   ├── generate.py    <- script that helps generate data
    │   └── instances.py   <- simple script to activate AWS EC2 instance
    │   
```
## Data

After using a Kaggle dataset that yielded bad results, I decided to generate my own dataset using images of my own hands. The code under /src called "generate.py" took images of my hand and saved it to generate a dataset.

There are 2900 images total: 100 images per gesture, with a total of 29 gestures (A-Z, Space, Backspace, and "Nothing")

## Future Works

I plan on converting the Python script into an iPhone app that achieves the same: translate live gesture into English text. Although it doesn't have an actual video chat capabilities, it will allow for more data collection and increase the model's robustness.
