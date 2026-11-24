"""Average saved OOF/test predictions into a new prediction set: python blend.py TASK OUT_NAME name1 name2 ..."""
import sys
import numpy as np
from common import PRED, save_pred
import os

task, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
ld = lambda n, s: np.load(os.path.join(PRED, f't{task}_{n}_{s}.npy'))
save_pred(task, out, np.mean([ld(n, 'test') for n in names], 0), np.mean([ld(n, 'oof') for n in names], 0))
