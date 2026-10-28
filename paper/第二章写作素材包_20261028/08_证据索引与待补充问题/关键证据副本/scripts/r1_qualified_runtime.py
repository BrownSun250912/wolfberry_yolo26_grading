"""Opt-in semantic-parity optimization; immutable old runtime remains intact."""
import numpy as np
from r1_research_runtime import ResearchGrader


class QualifiedOnlyGrader(ResearchGrader):
    def classify(self,image,masks,baseline):
        base=np.asarray(baseline,dtype=np.int64);use=np.flatnonzero(base!=0)
        if len(base)!=len(masks):raise ValueError('Class/mask count differs')
        labels=base.copy();score=np.full(len(base),np.nan);stem=np.full(len(base),np.nan)
        if len(use):
            cls,ss,sp=super().classify(image,[masks[i] for i in use],base[use])
            labels[use]=cls;score[use]=ss;stem[use]=sp
        return labels,score,stem


def shared_grader(original):
    """Share immutable eval-mode weights for paired checks, not a new model."""
    result=QualifiedOnlyGrader.__new__(QualifiedOnlyGrader);result.__dict__.update(original.__dict__)
    return result
