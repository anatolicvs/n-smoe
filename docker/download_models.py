import torchvision.models as models

def download_model(model_func, *args, **kwargs):
    model = model_func(*args, **kwargs)
    model.eval()

def main():
    # Classification models
    download_model(models.alexnet, pretrained=True)
    download_model(models.vgg16, pretrained=True)
    download_model(models.vgg16_bn, pretrained=True)
    download_model(models.vgg19, pretrained=True)
    download_model(models.vgg19_bn, pretrained=True)
    download_model(models.resnet18, pretrained=True)
    download_model(models.resnet34, pretrained=True)
    download_model(models.resnet50, pretrained=True)
    download_model(models.resnet101, pretrained=True)
    download_model(models.resnet152, pretrained=True)
    download_model(models.squeezenet1_0, pretrained=True)
    download_model(models.squeezenet1_1, pretrained=True)
    download_model(models.densenet121, pretrained=True)
    download_model(models.densenet169, pretrained=True)
    download_model(models.densenet201, pretrained=True)
    download_model(models.densenet161, pretrained=True)
    download_model(models.inception_v3, pretrained=True)
    download_model(models.googlenet, pretrained=True)
    download_model(models.shufflenet_v2_x1_0, pretrained=True)
    download_model(models.shufflenet_v2_x0_5, pretrained=True)
    download_model(models.mobilenet_v2, pretrained=True)
    download_model(models.mnasnet0_5, pretrained=True)
    download_model(models.mnasnet1_0, pretrained=True)
    
    # Segmentation models
    download_model(models.segmentation.fcn_resnet50, pretrained=True)
    download_model(models.segmentation.fcn_resnet101, pretrained=True)
    download_model(models.segmentation.deeplabv3_resnet50, pretrained=True)
    download_model(models.segmentation.deeplabv3_resnet101, pretrained=True)
    download_model(models.segmentation.lraspp_mobilenet_v3_large, pretrained=True)

    # Detection models
    download_model(models.detection.fasterrcnn_resnet50_fpn, pretrained=True)
    download_model(models.detection.maskrcnn_resnet50_fpn, pretrained=True)
    download_model(models.detection.keypointrcnn_resnet50_fpn, pretrained=True)
    
    # ... add more models as needed ...

if __name__ == "__main__":
    main()
