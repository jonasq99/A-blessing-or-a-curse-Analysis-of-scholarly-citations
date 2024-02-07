
import pandas as pd
from sklearn.metrics import f1_score
from utils import create_data, zero_shot
from tqdm import tqdm

df_dict = create_data(70, 30)

def filter_label(dataframes_dict: dict[pd.DataFrame], label: int) -> pd.DataFrame:
    # Create an empty list to store filtered DataFrames
    filtered_dataframes = []
    
    for key, df in dataframes_dict.items():
        if 'Label' in df.columns:
            filtered_df = df[df['Label'] == label]
            
            filtered_dataframes.append(filtered_df)
    
    result_df = pd.concat(filtered_dataframes, ignore_index=True)
    
    return result_df

def sample_data(dataframe):
    sampled_df = dataframe.sample(n=100, random_state=42) 
    return sampled_df


opinionated_data = filter_label(df_dict, 1)
neutral_data = sample_data(filter_label(df_dict, 0))

df = pd.concat([opinionated_data, neutral_data], ignore_index=True)

def get_precictions(df):
    predictions = []
    for i in tqdm(range(len(df))):
        name = df["Authors"].iloc[i]
        title = df["Title"].iloc[i]
        context = df["context"].iloc[i]
        footnote = df["footnote_text"].iloc[i]
        #pred = context_sentiment(context)
        pred = zero_shot(name, title, context,footnote)
        
        while pred != "0" and pred != "1":
            print("Retrying prediction...")
            pred = zero_shot(name, title, context,footnote)
        
        predictions.append(pred)
        predictions = [int(i) for i in predictions]
                
    return predictions

predictions = get_precictions(df)

labels = df["Label"].tolist()

def calculate_accuracy_per_label(predictions, labels, label_value):
    # Create a boolean array indicating whether the label matches the specified value
    label_matches = [l == label_value for l in labels]
  
    # Extract predictions for instances where the label matches the specified value
    matched_predictions = [p for i, p in enumerate(predictions) if label_matches[i]]
        
    return sum(matched_predictions)/100 if label_value == 1 else (len(matched_predictions) - sum(matched_predictions))/100
   
f1 = f1_score(predictions, labels)
accuracy_label_0 = calculate_accuracy_per_label(predictions, labels, label_value=0)
accuracy_label_1 = calculate_accuracy_per_label(predictions, labels, label_value=1)

print("F1 score:", f1)
print("Accuracy for label 0:", accuracy_label_0)
print("Accuracy for label 1:", accuracy_label_1)




