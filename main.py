from train import *
from video import predict_on_ID, predict_on_OOD


train(epochs=5)
predict_on_ID(confidence=0.5)
finetunning(epochs=5)
predict_on_OOD(confidence=0.5, visualize=True)