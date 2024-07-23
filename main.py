from train import *
from video import predict_on_ID, predict_on_OOD
from consts import CONFIDENCE_LEVEL

train(epochs=7)
predict_on_ID(confidence=CONFIDENCE_LEVEL)
finetunning(epochs=7)
predict_on_OOD(confidence=CONFIDENCE_LEVEL)