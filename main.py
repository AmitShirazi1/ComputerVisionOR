from train import train
from generate_pseudo_labels import predict_on_ID
from finetune import finetunning

train(epochs=7)
predict_on_ID(confidence=0)
finetunning()