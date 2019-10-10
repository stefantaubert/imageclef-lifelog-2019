import numpy as np
from src.models.pooling.Model import Model
from src.models.pooling.Model_opts import *

from experiments.opt_keys import opt_data_weights

param_order = {
    opt_use_seg: 1,
    opt_img_selection: 1,
    opt_repr_selection: 1,
    opt_s1_min_imgs: 1,
    opt_merge_dist: 1,
    opt_s1_threshold: 1,
    opt_query_src: 1,
    opt_threshold: 2,
    opt_optimize_labels: 2,
    opt_use_idf: 3,
    opt_idf_boosting_threshold: 3,
    opt_intensify_factor_m: 3,
    opt_intensify_factor_p: 3,
    opt_ceiling: 3,
    opt_comp_use_weights: 4,
    opt_comp_method: 4,
    opt_use_tokenclustering: 4,
    opt_tokenclustering_comp_method: 4,
    opt_data_weights: 4,
    opt_subm_imgs_per_day: 5,
    opt_subm_imgs_per_day_only_on_recall: 5,
}

class ExperimentModel(Model):

    def __init__(self, opts):
        self.reset_updates()
        self.__step1_index__ = 0
        self.__step2_index__ = 1
        self.__step3_index__ = 2
        self.__step4_index__ = 3
        self.__step5_index__ = 4
        self.methods_to_opts = {
            opt_use_seg: self.change_use_seg,
            opt_query_src: self.change_query_src,
            opt_img_selection: self.change_img_selection,
            opt_repr_selection: self.change_repr_selection,
            opt_s1_min_imgs: self.change_s1_min_imgs,
            opt_merge_dist: self.change_merge_dist,
            opt_s1_threshold: self.change_s1_threshold,
            opt_threshold: self.change_threshold,
            opt_optimize_labels: self.change_optimize_labels,
            opt_use_idf: self.change_use_idf,
            opt_idf_boosting_threshold: self.change_idf_boosting_threshold,
            opt_intensify_factor_m: self.change_intensify_factor_m,
            opt_intensify_factor_p: self.change_intensify_factor_p,
            opt_ceiling: self.change_ceiling,
            opt_comp_use_weights: self.change_comp_use_weights,
            opt_comp_method: self.change_comp_method,
            opt_data_weights: self.change_data_weights,
            opt_subm_imgs_per_day: self.change_subm_imgs_per_day,
            opt_subm_imgs_per_day_only_on_recall: self.change_subm_imgs_per_day_only_on_recall,
            opt_use_tokenclustering: self.change_use_tokenclustering,
            opt_tokenclustering_comp_method: self.change_tokenclustering_comp_method,
        }

        self.reset()
        return super().__init__(opts)

    def reset(self):
        self.set_update(self.__step1_index__)

    def change_opt(self, opt, value):
        assert opt in self.methods_to_opts
        change_method = self.methods_to_opts[opt]
        change_method(value)

    def change_query_src(self, query_src):
        assert opt_query_src in self.opts[opt_model]
        if self.opts[opt_model][opt_query_src] != query_src:
            self.opts[opt_model][opt_query_src] = query_src
            self.set_update(self.__step1_index__)

    def change_use_seg(self, use_seg):
        assert opt_use_seg in self.opts[opt_model]
        if self.opts[opt_model][opt_use_seg] != use_seg:
            self.opts[opt_model][opt_use_seg] = use_seg
            self.set_update(self.__step1_index__)

    def change_img_selection(self, img_selection):
        if self.opts[opt_segmentation][opt_img_selection] != img_selection:
            self.opts[opt_segmentation][opt_img_selection] = img_selection
            self.set_update(self.__step1_index__)

    def change_repr_selection(self, repr_selection):
        if self.opts[opt_segmentation][opt_repr_selection] != repr_selection:
            self.opts[opt_segmentation][opt_repr_selection] = repr_selection
            self.set_update(self.__step1_index__)

    def change_s1_min_imgs(self, s1_min_imgs):
        if self.opts[opt_segmentation][opt_s1_min_imgs] != s1_min_imgs:
            self.opts[opt_segmentation][opt_s1_min_imgs] = s1_min_imgs
            self.set_update(self.__step1_index__)

    def change_merge_dist(self, merge_dist):
        if self.opts[opt_segmentation][opt_merge_dist] != merge_dist:
            self.opts[opt_segmentation][opt_merge_dist] = merge_dist
            self.set_update(self.__step1_index__)

    def change_s1_threshold(self, s1_threshold):
        if self.opts[opt_segmentation][opt_s1_threshold] != s1_threshold:
            self.opts[opt_segmentation][opt_s1_threshold] = s1_threshold
            self.set_update(self.__step1_index__)

    def change_threshold(self, threshold):
        data_key = list(self.opts[opt_data].keys())[0]
        if self.opts[opt_data][data_key][opt_threshold] != threshold:
            self.opts[opt_data][data_key][opt_threshold] = threshold
            self.set_update(self.__step2_index__)

    def change_optimize_labels(self, optimize_labels: bool):
        assert opt_optimize_labels in self.opts[opt_model]
        if self.opts[opt_model][opt_optimize_labels] != optimize_labels:
            self.opts[opt_model][opt_optimize_labels] = optimize_labels
            self.set_update(self.__step2_index__)

    def change_use_idf(self, use_idf):
        data_key = list(self.opts[opt_data].keys())[0]
        if self.opts[opt_data][data_key][opt_use_idf] != use_idf:
            self.opts[opt_data][data_key][opt_use_idf] = use_idf
            self.set_update(self.__step3_index__)

    def change_idf_boosting_threshold(self, idf_boosting_threshold):
        data_key = list(self.opts[opt_data].keys())[0]
        if self.opts[opt_data][data_key][opt_idf_boosting_threshold] != idf_boosting_threshold:
            self.opts[opt_data][data_key][opt_idf_boosting_threshold] = idf_boosting_threshold
            self.set_update(self.__step3_index__)

    def change_intensify_factor_m(self, intensify_factor_m):
        data_key = list(self.opts[opt_data].keys())[0]
        if self.opts[opt_data][data_key][opt_intensify_factor_m] != intensify_factor_m:
            self.opts[opt_data][data_key][opt_intensify_factor_m] = intensify_factor_m
            self.set_update(self.__step3_index__)

    def change_intensify_factor_p(self, intensify_factor_p):
        data_key = list(self.opts[opt_data].keys())[0]
        if self.opts[opt_data][data_key][opt_intensify_factor_p] != intensify_factor_p:
            self.opts[opt_data][data_key][opt_intensify_factor_p] = intensify_factor_p
            self.set_update(self.__step3_index__)

    def change_ceiling(self, ceiling):
        data_key = list(self.opts[opt_data].keys())[0]
        if self.opts[opt_data][data_key][opt_ceiling] != ceiling:
            self.opts[opt_data][data_key][opt_ceiling] = ceiling
            self.set_update(self.__step3_index__)

    def change_comp_method(self, comp_method):
        if self.opts[opt_model][opt_comp_method] != comp_method:
            self.opts[opt_model][opt_comp_method] = comp_method
            self.set_update(self.__step4_index__)

    def change_comp_use_weights(self, comp_use_weights):
        if self.opts[opt_model][opt_comp_use_weights] != comp_use_weights:
            self.opts[opt_model][opt_comp_use_weights] = comp_use_weights
            self.set_update(self.__step4_index__)

    def change_use_tokenclustering(self, use_tokenclustering):
        assert opt_use_tokenclustering in self.opts[opt_model]
        if self.opts[opt_model][opt_use_tokenclustering] != use_tokenclustering:
            self.opts[opt_model][opt_use_tokenclustering] = use_tokenclustering
            self.set_update(self.__step4_index__)

    def change_tokenclustering_comp_method(self, tokenclustering_comp_method):
        assert opt_tokenclustering_comp_method in self.opts[opt_tokenclustering]
        if self.opts[opt_tokenclustering][opt_tokenclustering_comp_method] != tokenclustering_comp_method:
            self.opts[opt_tokenclustering][opt_tokenclustering_comp_method] = tokenclustering_comp_method
            self.set_update(self.__step4_index__)

    def change_data_weights(self, data_weights):
        changed_any_weight = False
        for k in self.opts[opt_data]:
            assert k in self.opts[opt_data]
            assert k in data_weights
            if self.opts[opt_data][k][opt_weight] != data_weights[k]:
                self.opts[opt_data][k][opt_weight] = data_weights[k]
                changed_any_weight = True
                
        if changed_any_weight:
            self.set_update(self.__step4_index__)
        
    def change_subm_imgs_per_day(self, subm_imgs_per_day):
        if self.opts[opt_model][opt_subm_imgs_per_day] != subm_imgs_per_day:
            self.opts[opt_model][opt_subm_imgs_per_day] = subm_imgs_per_day
            self.set_update(self.__step5_index__)

    def change_subm_imgs_per_day_only_on_recall(self, subm_imgs_per_day_only_on_recall):
        if self.opts[opt_model][opt_subm_imgs_per_day_only_on_recall] != subm_imgs_per_day_only_on_recall:
            self.opts[opt_model][opt_subm_imgs_per_day_only_on_recall] = subm_imgs_per_day_only_on_recall
            self.set_update(self.__step5_index__)

    def reset_updates(self):
        self.__updates__ = [False] * 5

    def set_update(self, index):
        self.__updates__[index] = True

    def need_update(self, index):
        for i in self.__updates__[:(index + 1)]:
            if i: return True
        return False

    def step1(self):
        if self.need_update(self.__step1_index__):
            super(ExperimentModel, self).fit_preprocessing()

    def step2(self):
        if self.need_update(self.__step2_index__):
            super(ExperimentModel, self).fit_data_extraction()
            super(ExperimentModel, self).fit_comparers()

    def step3(self):
        if self.need_update(self.__step3_index__):
            super(ExperimentModel, self).compare()

    def step4(self):
        if self.need_update(self.__step4_index__):
            super(ExperimentModel, self).predict()

    def step5(self):
        if self.need_update(self.__step5_index__):
            super(ExperimentModel, self).postprocess()

    def run(self):
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        self.reset_updates()
        return self.subm