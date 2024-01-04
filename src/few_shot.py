#import os
#import json
#import pandas as pd
#from openai import OpenAI
#from text_extraction import file_finder, TextExtraction
#from dotenv import load_dotenv
#from pathlib import Path
from utils import get_completion, get_completion_from_messages, create_data


data = create_data()
print(data.keys())
