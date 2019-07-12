# starts new transformations after histogram extraction (usefull because image loading takes long time)
opt_soft_clear = "soft_clear"

# clears the cache, necessary if the user is changed
opt_clear_cache = "clear_cache"

opt_multiprocessing = "multiprocessing"
opt_usr = "usr"

# copy clusters to filesystem to watch the results of the segmentation
opt_copy_clusters = "copy_clusters"
opt_img_selection = "img_selection"
opt_s1_min_imgs = "s1_min_imgs"

# merge segments which are n images apart (usefull for reducing the total amount of segments)
# value -1 says, no merging is applied
opt_merge_dist = "merge_dist"

opt_linkage_method = "linkage_method"
opt_linkage_metric = "linkage_metric"
opt_fcluster_criterion = "fcluster_criterion"
opt_s1_threshold = "s1_threshold"

linkage_method_average = "average"
linkage_metric_euclidean = "euclidean"
fcluster_criterion_distance = "distance"
