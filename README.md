# MachineLearning
    

1. `preprocessing.py` is a script that can be used to preprocess data for use in the training model
    - usage: ```python preprocessing.py raw.csv```, where raw.csv contains the raw data to be used
    
2. `catnn.py` is a script that can be used to train a neural net that predicts whether a company's stock will increase by 10% or decrease by 10% in a quarter. The script saves the resulting model in a folder named 'nnmodel' so that it can later be used for predictions in another script.
    - usage: ```python catnn.py train.csv```, where data.csv contains the training data to be used
    
3. `predictnn.py` is a script that makes a outputs using the neural network model produced by `catnn.py`
    - usage: ```python predict.py```
