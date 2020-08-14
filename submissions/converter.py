import pandas as pd

def subm_to_df(subm):
    data = []
    
    for key in sorted(subm.keys()):
        top_imgs = subm[key]

        for img_id, score in top_imgs.items():
            row = [key, img_id, score]
            data.append(row)

    return pd.DataFrame(columns=["topic_id", "image_id", "confidence_score"], data=data)