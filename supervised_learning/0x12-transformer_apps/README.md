# 0x12. Transformer Applications

## Table of Content
* [Description](#description)
* [Learning Objectives](#learning-objectives)
* [Files Description](#files-description)
* [Authors](#authors)

## Description

Transformers application.

## Learning Objectives
### General


- How to use Transformers for Machine Translation.
- How to write a custom train/test loop in Keras.
- How to use Tensorflow Datasets.




## Files Description

[0-dataset.py](0-dataset.py) - Loads and preps a dataset for machine translation.

[1-dataset.py](1-dataset.py) - Create the instance method `def encode(self, pt, en):` that encodes a translation into tokens.

[2-dataset.py](2-dataset.py) - Method that acts as a tensorflow wrapper for the encode instance method.

[3-dataset.py](3-dataset.py) - Update the class constructor `def __init__(self, batch_size, max_len):`

[4-create_masks.py](4-create_masks.py) - creates all masks for training/validation:

[5-train.py](5-train.py) - creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset.

[5-transformer.py](5-transformer.py) - transformer implementation.

## Authors
* Diego Gomez- [Linkedin](https://www.linkedin.com/in/diego-g%C3%B3mez-8861b61a1/) / [Twitter](https://twitter.com/dagomez2530)
