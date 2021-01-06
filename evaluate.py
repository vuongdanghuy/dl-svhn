import numpy as np
import pandas as pd
import config as cfg
from scoring import mAP

mAP(cfg.TEST_LABEL_FILE, './output/pred.csv', plot=True)