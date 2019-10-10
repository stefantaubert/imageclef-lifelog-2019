import os
import sys

from src.models.pooling.Model import Model
from src.models.pooling.Model_opts import *
from experiments.evaluation.main import evaluate
from submissions.show_in_fs import show_in_fs_ctx
from src.io.ReadingContext import ReadingContext
from src.globals import ds_dev
from src.globals import ds_test
from src.globals import usr1
from submissions.creator import create_submission

gen_opts = {
    opt_general: {
        opt_usr: usr1,
        opt_mp: True,
        opt_ds: None,
    }
}

img_count_fs = 20

def run_on_dev(opts):
    model_name = os.path.basename(sys.argv[0])[:-3]
    print("Started", model_name, "on devset.")
    train_opts = dict(opts)
    train_opts.update(gen_opts)
    train_opts[opt_general][opt_ds] = ds_dev
    m = Model(train_opts)
    subm = m.run()
    _, eval_res_df = evaluate(subm, Xs=[5, 10, 20, 30, 40, 50])
    f1_10 = float(eval_res_df.loc[(eval_res_df['X'] == 10) & (eval_res_df['topic_id'] == 'all')]["F1@X"])
    print("F1@10:", str(f1_10))

    fs_dir = show_in_fs_ctx(subm, m.ctx, model_name, img_count_fs)
    print("Written top {} images per dev topic to:".format(str(img_count_fs)), fs_dir)

def run_on_test(opts):
    model_name = os.path.basename(sys.argv[0])[:-3]
    print("Started", model_name, "on testset.")
    test_opts = dict(opts)
    test_opts.update(gen_opts)
    test_opts[opt_general][opt_ds] = ds_test
    m = Model(test_opts)
    subm = m.run()

    fs_dir = show_in_fs_ctx(subm, m.ctx, model_name, img_count_fs)
    print("Written top {} images per test topic to:".format(str(img_count_fs)), fs_dir)

    file_name = create_submission(subm, "LMRT_TUC_MI_Stefan_Taubert_" + model_name)
    print("exported test submission to", file_name)
