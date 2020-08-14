def gt_to_dict(gt):
    result = { x: { } for x in set(gt["topic_id"]) }
    
    for index, row in gt.iterrows():
        top = row["topic_id"]
        img = row["image_id"]
        cluster = row["topic_cluster_id"]
        cluster_added = cluster in result[top]

        if not cluster_added:
            result[top][cluster] = []
        
        result[top][cluster].append(img)
    
    return result
