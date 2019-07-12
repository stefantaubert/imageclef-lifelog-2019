from pdtransform import NoFitMixin

class MinMaxPrinterTransformer(NoFitMixin):
    def transform(self, submission):
        max_score = max(submission.values())
        min_score = min(submission.values())
        print("min: {min}, max: {max}, diff: {diff}".format(min=str(min_score), max=str(max_score), diff=str(max_score-min_score)))
        return submission