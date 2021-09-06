# ELVis (Easy Language-and-VISion)

ELVis is a PyTorch based library that aims to be an extensible tool to implement multi-gpu, mixed-precision Deep Learning experiments using yaml configurable file for clean setup.

## Installation

```
pip install requirements.txt
cd elvis
pip install -e .
```

## Features
- distributed training
- mixed-precision training
- yaml configuration file
- logging
- Language-and-Vision architectures

## Interfaces

The library offers different interfaces for neural architectures, trainers and data.

- __Trainer__: setup the training process from the yaml configuration file
- __Meta-architecture__: define the training pipeline for different tasks (e.g., VQA, Retrieval)
- __Model__: the specific model to be instantiated and trained for the task. Each model has different interfaces for different tasks. Each interface define how to read and format data for the trainer (e.g., create a sequence for a transformer, resize images for a CNN). Each interface in turn defines two methods:
   - __worker_fn__: defines the data reading for each worker of the PyTorch's DataLoader
   - __collate_fn__: defines how to batch data.
- __Dataset__: defines the dataset

![Diagram](/pictures/diagram.png)

## Usage
An example of usage is provided in ![example](/example) folder.
