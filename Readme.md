# Abstract

Project employs deep learning on DDoS datasets for predictive models, enhancing cybersecurity by preemptively identifying and mitigating attacks.

# Code

## Dataset used

Took the dataset from

https://data.mendeley.com/datasets/ssnc74xm6r/1

Download the dataset and add it in the folder before running the below python files.

## Spliting dataset into 3 parts

split_dataset.py is used to split the data set into training, validation and testing.

For example, if you have 1000 samples in your dataset, you might split it as follows:

700 samples for training (70%) 150 samples for validation (15%) 150 samples for testing (15%)

## Preprocessing the dataset

preprocess.py

We do hot encoding, removing the null values and encoding labels that can be encoded.

## Running the model

deepl.py

We run and test out the model.

## Output



