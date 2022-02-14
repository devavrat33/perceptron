
from fileinput import filename
import pandas as pd
import numpy as np
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot

def main(data, eta, epochs, filename, plotname):
      

      df = pd.DataFrame(data)
      print(f'df = {df}')
      X, y = prepare_data(df)

      
      model_AND = Perceptron(eta=eta, epochs=epochs)
      model_AND.fit(X,y)
      _  = model_AND.total_loss()

      save_model(model_AND, file_name=filename)
      save_plot(df, plotname, model_AND)


if __name__ == '__main__':
      AND = {
            'x1': [0,0,1,1],
            'x2': [0,1,0,1],
            'y' : [1,1,1,0]
      }
      ETA = 0.3
      EPOCHS = 10
 
      main(data=AND, eta = ETA, epochs = EPOCHS, filename='or.model', plotname='or.png' )