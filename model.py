import segmentation_models_pytorch as smp

def get_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,   # VERY IMPORTANT
        classes=1
    )
    return model