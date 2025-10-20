# ECG_Capstone
My Capstone ECG Project

## Pre Requisites:

This was created using python 3.11

To run, install the requirements.txt run:

pip install -r requirements.txt

OR Use the environment.yml file to create a conda environment 

conda env create -f environment.yml

## Running:

The order in which to run the files to get results is the following:

First run the download_data.py This will download the necessary data, Make sure you have at least 3GB free for the download. You will need a total of 16GB for the whole project to run.

Next Navigate to MITBIH\model_arxiv-1805-00794 and in that folder run original_model-from-arxiv-1805-00794.ipynb from start to end. This will leave you with the baseline Human model

Sidebar Here, for comparision there is also a binary model inside binary_model folder, this is to show that the method I use for the cainine is also applicable here.

After creating the human model, then move accorss to the Cainine folder Navigate to: Cainine\DataPreProcessing and run:

First: enhanced_preprocessing.py

Second: create_train_test_split.py

This will process that data and create our test and train datasets.

Navigate to Cainine\Transfer of Learning and then run:

First: transfer_cainine_v2.py this classifies as a binary Normal or Arrhythmia

Second: train_subclassifier_s_vs_v.py this is the subclassifier to classify the arrhythmias

## That is the training done.

Now run test_hierarchical_canine.py This will test against the test set we created earlier

And the results you see are now the results from the model on the cainine data.

