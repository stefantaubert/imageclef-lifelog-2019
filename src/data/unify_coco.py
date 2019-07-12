def unify_coco(coco):
    """
    Unifies concept labels. Transforms the labels to the official labels.
    - some use 'traffic light' and others 'traffic_light'
    """
    coco = coco.replace('_', ' ')
    if coco == "tvmonitor":
        return "tv"
    elif coco == "aeroplane":
        return "airplane"
    elif coco == "diningtable":
        return "dining table"
    elif coco == "sofa":
        return "couch"
    elif coco == "motorbike":
        return "motorcycle"
    elif coco == "pottedplant":
        return "potted plant"
    else:
        return coco
