import sys
import warnings

from src.information_extraction import information_extraction, extract_citations, tagger_information_extraction

warnings.filterwarnings("ignore")


if len(sys.argv) < 3:
    print("""Missing information_extraction_method argument. 
    
    Usage: python main.py tagger_name filename
    
    tagger_name: anystyle, regex, tagger
    filename: the name of the file in the folder all_data_articles
    """)
    sys.exit(1)

information_extraction_method = sys.argv[1]
filename = sys.argv[2]

"""
anystyle -> information_extraction
regex -> extract_citations
tagger -> tagger_information_extraction
"""

extraction = None

if information_extraction_method == "anystyle":
    extraction = information_extraction(filename, "all_data_articles", "src")
elif information_extraction_method == "regex":
    extraction = extract_citations(filename, "all_data_articles")
elif information_extraction_method == "tagger":
    from flair.nn import Classifier

    tagger = Classifier.load('ner-ontonotes-large')
    extraction = tagger_information_extraction(filename, tagger, "all_data_articles")
else:
    raise ValueError("Invalid information_extraction_method")

print(extraction)
