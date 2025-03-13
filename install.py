import os
import sys


CUSTOM_NODES_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CUSTOM_NODES_PATH, "configs")
WEIGHTS_PATH = os.path.join(CUSTOM_NODES_PATH, "weights")

sys.path.append(CUSTOM_NODES_PATH)



