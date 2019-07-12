# General Options
opt_general = "general"
opt_usr = "usr"
opt_ds = "dataset"
opt_mp = "multiprocessing"

# Model Options

opt_model = "model"
opt_subm_imgs_per_day = "subm_img_per_day"
opt_subm_imgs_per_day_only_on_recall = "subm_imgs_per_day_only_on_recall"
opt_use_seg = "use_segmentation"
opt_use_tokenclustering = "use_token_clusters"
opt_comp_method = "comparison_method"
opt_comp_use_weights = "comparison_use_weights"
opt_query_src = "query_src"
opt_optimize_labels = "optimize_labels"

# Token Clustering

opt_tokenclustering = "tokenclustering"
opt_tokenclustering_threshold = "tc_threshold"
opt_tokenclustering_comp_method = "comp_method"

# Segmentation Options

opt_segmentation = "segmetation"
opt_img_selection = "img_selection"
opt_repr_selection = "repr_selection"
opt_s1_min_imgs = "s1_min_imgs"
opt_merge_dist = "merge_dist"
opt_s1_threshold = "s1_threshold"

# Data Options

opt_data = "data"
opt_weight = "weight"
opt_threshold = "threshold"
opt_use_idf = "use_idf"
opt_idf_boosting_threshold = "idf_boosting_threshold"
opt_intensify_factor_m = "intensify_factor_m"
opt_intensify_factor_p = "intensify_factor_p"
opt_ceiling = "ceiling"

# Query Sources

query_src_title = "title"
query_src_desc = "description"
query_src_narrative = "narrative"
query_src_manual = "manual"

""" Takes title, description and narrative for the query. """
query_src_all = "all"

## Comparison Methods

""" Calculates the mean of all similarities for all tokens. """
comp_method_mean = "mean"

""" Takes the maximal similarity per labeltype of any token and calculates the mean (rowmax). """
comp_method_datamax = "datamax"

""" Takes the maximal similarity per token of any labeltype and calculates the mean (colmax). """
comp_method_tokenmax = "tokenmax"

### opt_tokenclustering_comp_method
""" Takes the mean of the similarities of each cluster. """
comp_method_clusters_mean = "mean"
""" Takes only the maximum of the similarities over all clusters """
comp_method_clusters_max = "max"

# # Segmentation options

img_selection_first = "first"
img_selection_last = "last"
img_selection_all = "all"
repr_selection_first = "first"
repr_selection_last = "last"
repr_selection_all = "all"