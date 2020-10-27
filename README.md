# MachineLearning

1. `train.py` is a script that can be used to train a logistic regression model that predicts whether a company's stock will increase by 10% in 1 year. The script saves the resulting model in a file named 'model.sav' so that it can later be used for predictions in another script.
    - usage: ```python train.py data.csv```, where data.csv contains the training data to be used
    
2. `predict.py` is a script that makes a prediction using the logistic regression model produced by `train.py`
    - usage: ```python predict.py model.sav input.csv```, where model.sav is the file containing the regression model, and input.csv contains the input variables to use for the prediction
    
**TODO:** Write logic to assess the performance of the trained model, rewrite scripts so that they are easily compatible with REACT, try several different types of models
