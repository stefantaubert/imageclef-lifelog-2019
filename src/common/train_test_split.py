from sklearn.model_selection import train_test_split

def train_test_split_list(imgs: list, test_size: float, shuffle: bool, seed: int):
    return train_test_split(imgs, test_size=test_size, shuffle=shuffle, random_state=seed)

def train_test_split_gt(gt: dict, test_size: float, shuffle: bool, seed: int):
    train = {}
    test = {}
    for top in gt.keys():
        train[top] = []
        test[top] = []
        for cl in gt[top].keys():
            imgs = gt[top][cl]
            tr, te = train_test_split_list(imgs, test_size, shuffle, seed)
            train[top].extend(tr)
            test[top].extend(te)
        # images in multiple clusters of same topic are counted only once
        train[top] = list(train[top])
        test[top] = list(test[top])
    return (train, test)

if __name__ == "__main__":
    from src.io.reader import read_gt
    from src.evaluation.gt_converter import gt_to_dict
    gt = read_gt()
    gt = gt_to_dict(gt)
    x1_train, x1_test = train_test_split_gt(gt, 0.2, True, 0)
    print(x1_train[1][:10])
    print(x1_test[1][:10])