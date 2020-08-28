### Class in Production
import pickle

class VectTweet(object):
  def __init__(self):
    self.vect = pickle.load(open('vect.pkl','rb'))
    
  def data_preparation(self, df):
    
    # Vectorizing
    
    X = self.vect.transform(df)
    
    return X