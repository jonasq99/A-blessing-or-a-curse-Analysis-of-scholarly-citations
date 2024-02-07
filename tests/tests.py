import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

from src.utils import get_completion
from src.data_creator import create_data
from src.synthetic_data_generation import create_opinionated
