import os
import matplotlib

IS_IPYTHON = 'inline' in matplotlib.get_backend()
DATA_PATH = os.path.join(os.getcwd(), 'models')
POLICY_NET_FILE = os.path.join(DATA_PATH, 'policy_network.pth')
TARGET_NET_FILE = os.path.join(DATA_PATH, 'target_network.pth')