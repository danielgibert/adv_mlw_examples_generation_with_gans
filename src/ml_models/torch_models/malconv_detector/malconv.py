from keras.models import load_model
import os

def restore_model(model_weights_filepath:str):
    print("Restoring malconv.h5 from disk for continuation training...")
    basemodel = load_model(model_weights_filepath)
    basemodel.summary()
    return basemodel