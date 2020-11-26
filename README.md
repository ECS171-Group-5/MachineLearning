# MachineLearning

1. `log.py` is a script that can be used to train a logistic regression model that predicts whether a company's stock will increase by 10% in a quarter. The script saves the resulting model in a file named 'model.sav' so that it can later be used for predictions in another script.
    - usage: ```python log.py data.csv```, where data.csv contains the training data to be used
    
2. `catnn.py` is a script that can be used to train a neural net that predicts whether a company's stock will increase by 10% or decrease by 10% in a quarter. The script saves the resulting model in a file named 'model.sav' so that it can later be used for predictions in another script.
    - usage: ```python catnn.py data.csv```, where data.csv contains the training data to be used
    
3. `predictlog.py` is a script that makes a prediction using the logistic regression model produced by `log.py`
    - usage: ```python predict.py model.sav input.csv```, where model.sav is the file containing the regression model, and input.csv contains the input variables to use for the prediction
