from train import *
from video import predict_on_ID, predict_on_OOD


epochs = 10
conf_level = 0.5

train(epochs=epochs)
predict_on_ID(confidence=conf_level)
finetunning(epochs=epochs)
predict_on_OOD(confidence=conf_level, visualize=True)