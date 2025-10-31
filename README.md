# ECG_Capstone
My Capstone ECG Project

## Pre Requisites:

This was created using python 3.11

To run, install the requirements.txt run:

pip install -r requirements.txt

  Use the environment.yml file to create a conda environment 

conda env create -f environment.yml

## Running Human:

The order in which to run the files to get results is the following:

First run the download_data.py This will download the necessary data, Make sure you have at least 3GB free for the download. You will need a total of 16GB for the whole project to run.

Next Navigate to MITBIH\model_arxiv-1805-00794 and in that folder run original_model-from-arxiv-1805-00794.ipynb from start to end. This will leave you with the baseline Human model

Sidebar Here, for comparision there is also a binary model inside binary_model folder, this is to show that the method I use for the cainine is also applicable here.

## Preprocessing Canine:
Navigate to: Cainine\DataPreProcessing_Inter for Interpatient Data or Cainine\DataPreProcessing_Intra for Intra patient data and run:

### Intra:
First: enhanced_preprocessing.py

Second: create_train_test_split.py

This will process that data and create our test and train datasets.

### Inter:
Run enhanced_preprocessing.py

## Running Canine:
Navigate to Cainine\Transfer of Learning\01_baseline and then:

### Inter (Currently set)

First Run train_stage1_original.py

Second Run train_stage2.py

Third Run test_hierarchical.py

#### Threshold analysis:
If the threshold which is currently set for stage 1 is failing (currently set for Inter)

Run analyze_threshold.py 

This will give an output of a threshold which you can set in the test_hierarchical.py on line 54

### Intra (Not Currently set)

#### Update Values for Intra:

train_stage1_original.py Line 33 update "DataPreProcessing_Inter" to "DataPreProcessing_Intra"

train_stage2.py Line 44 "DataPreProcessing_Inter" to "DataPreProcessing_Intra"

test_hierarchical.py Line 44 "DataPreProcessing_Inter" to "DataPreProcessing_Intra"

#### Run:

First Run train_stage1_original.py

Second Run train_stage2.py

Third Run test_hierarchical.py

And the results you see are now the results from the model on the cainine data.

