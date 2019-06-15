# Sign Language Translator


## Overview

This project developed an application that can translate ASL alphabet into English alphabet and prints it on-screen.

## Project Design

Using a self-made dataset of 100 images each of 29 gestures (26 English alphabets + "Space", "Backspace", "Nothing"), I trained two architectures: VGG16 and MobileNet V1. Instead of training all the weights, the pre-weights were loaded from the ImageNet dataset and Transfer Learning was applied.

## Application

Using OpenCV, I coded a program in Python that takes live video where a gesture inside a box is recognized and printed on screen for ASL translation into English language.

## Tools

- Data Storage: AWS S3
- Data Visualization: TensorBoard
- Deep Learning: Keras w/ Tensorflow Backend
- Application: Python, OpenCV
- Presentation: Google Slides

## Data

After using a Kaggle dataset that yielded bad results, I decided to generate my own dataset using images of my own hands. The code under /src called "generate.py" took images of my hand and saved it to generate a dataset.

There are 2900 images total: 100 images per gesture, with a total of 29 gestures (A-Z, Space, Backspace, and "Nothing")

## Appendix
