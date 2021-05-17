try:
    from efficientnet_pytorch import EfficientNet
except:
    print(
        "Please install efficientnet_pytorch using 'pip install efficientnet_pytorch'"
    )


def efficientnet(num_classes, pretrained=False, name="efficientnet-b3"):
    if pretrained == True:
        return EfficientNet.from_pretrained(name, num_classes=num_classes)
    else:
        return EfficientNet.from_name(name, num_classes=num_classes)
