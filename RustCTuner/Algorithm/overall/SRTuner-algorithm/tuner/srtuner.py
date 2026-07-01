from SRTuner import SRTunerModule
from .common import Tuner

def flags_to_binary_list(flag_dict):
    return [1 if v else 0 for v in flag_dict.values()]

class SRTuner(Tuner):
    def __init__(self, search_space, evaluator, log_file):
        super().__init__(search_space, evaluator, log_file)

        self.mod = SRTunerModule(
            search_space=search_space,
            evaluator=evaluator,
            default_perf=self.default_perf,
            LOG_FILE=log_file
        )


    def generate_candidates(self, batch_size=1, enable_expansion=True):
        return self.mod.generate_candidates(batch_size=batch_size, enable_expansion=enable_expansion)


    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(opt_setting) for opt_setting in candidates]


    def reflect_feedback(self, perfs):
        self.mod.reflect_feedback(perfs)




    def tune(self, budget, batch_size=1, enable_expansion=True):

        # return self.mod.tune(budget, batch_size)
        #

        import time
        best_opt_setting, best_perf = None, float("inf")
        ts = [0.0]
        start = time.time()
        res = []
        seqs = []
        while ts[-1] < budget:
            candidates = self.generate_candidates(batch_size=batch_size, enable_expansion=enable_expansion)
            perfs = self.evaluate_candidates(candidates)
            self.reflect_feedback(perfs)

            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            

            now = time.time()
            ts.append(now - start)
            flag_seq = flags_to_binary_list(opt_setting)
            pref_over_o3 = self.default_perf / best_perf
            time_bias = self.evaluator.evaluate_default()
            seqs.append(flag_seq)
            res.append(pref_over_o3)
            best_result = max(res)
            best_seq = seqs[res.index(best_result)]
            ss = f"{round(ts[-1])}: cur-best {best_result}, cur-best-seq {best_seq}"

            log_file = getattr(self.mod, "LOG_FILE", getattr(self, "LOG_FILE", None))
            if log_file:
                from .common import write_log
                write_log(ss, log_file)

        return best_opt_setting, best_perf
