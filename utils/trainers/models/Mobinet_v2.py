from transformers import  MobileNetV2ForImageClassification


def Mobinet():
    model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    return model


