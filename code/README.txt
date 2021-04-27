This folder contains a folder named data, 1 python file for parser, and 7 python files for models. 


The data folder contains the original dataset, preprocessed datasets, GloVe embeddings, and python programs needed to process the datasets. This folder should not be touched. 


The parser python file is used for tokenization of URLs and is called during runtime. 


Make sure that you have installed the latest version of tensorflow, keras, and sklearn libraries. Other libraries needed include numpy and pandas. 


To run the models, simply use the following command:  


python3 model_name.py


The output should be 8 AUC-ROC scores, with the preceding 4 scores for validation (the entire training set) and the later 4 scores for the test set. 


Note that there might be discrepancies between output scores and the highest scores in the report as different machines may have subtle differences in model training or some files might have been updated after obtaining the best-performing hyperparameters.