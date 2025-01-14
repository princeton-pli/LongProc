import os
import pandas as pd
import string
import re
import csv

_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)
def _normalize_answer(s: str):
    """Lower text and remove punctuation, articles and extra whitespace."""
    ## Remove articles
    def remove_articles(text: str) -> str:
        return _ARTICLES_REGEX.sub(" ", text)
    ## Fix white spaces
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    ## Remove white spaces
    def remove_white_spaces(text: str) -> str:
        return text.replace(' ','')
    ## Remove punctuation
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    ## Lowercase the text
    def lower(text: str) -> str:
        return text.lower()
    
    return remove_white_spaces(white_space_fix(remove_articles(remove_punc(lower(s)))))

def _string_to_pd(text:[]) -> pd.DataFrame:
    ## Convert a list of string in TSV format to pandas dataframe
    try:
        csv_reader = csv.reader(text, delimiter='\t')
        
        data = list(csv_reader)
        ## Take the first line as the header
        header = data[0]
        ## Remove the header from the data
        data = data[1:]
        ## If the last line is not complete, it is likely to be an empty line or some end markers (e.g. "The above is my extracted csv" in the model response), so just remove it
        if len(data[-1]) != len(header):
            data = data[:-1]
        
        for idx,line in enumerate(data):
            if len(line) != len(header):
                ## If one of the line is not complete or has extra columns than expected , fill it with N/A
                print(f'Line {idx} is not complete or has extra columns:',line)
                data[idx] = ['N/A' for _ in range(len(header))]
        
        df = pd.DataFrame(data,columns=header)
        df = df.fillna('N/A')
        ## Normalize all answers
        for column in df.columns:
            df[column] = df[column].astype(str).apply(_normalize_answer)
        return df
    except Exception as e:
        print(data)
        print('Error in converting string in TSV format to pandas dataframe')
        print(e)
        
        return None

def  evaluate_html_to_csv_compute_metrics(prediction : str, groundtruth: str) -> dict:
    try:
        ## Convert the groundtruth pandas dataframe
        gt_text = groundtruth.lstrip().rstrip().split('\n') 
        gt_df = _string_to_pd(gt_text)
        ## Convert the prediction to pandas dataframe
        pred_text = prediction.lstrip().rstrip().split('\n')
        pred_df = _string_to_pd(pred_text)
        ## Compute the precision score
        precision = 0
        for i in range(len(pred_df.index)):
            ## For each row in the prediction, check if it is in the groundtruth dataframe
            ## Note that for precision, we don't care if the prediction follows exactly the order of the groundtruth
            corr = (gt_df.eq(pred_df.iloc[i].values)).all(axis=1).any()
            precision+=corr
            
        precision/= len(pred_df.index)

        ## Compute the UNORDERED recall score
        recall = 0
        for i in range(len(gt_df.index)):
            ## For each row in the ground truth, check if it is in the prediction dataframe
            corr = (pred_df.eq(gt_df.iloc[i].values)).all(axis=1).any()
            recall+=corr
        recall/= len(gt_df.index)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        return {'precision': precision, 'recall': recall, 'f1': f1,"error": None}
    except Exception as e:
        print('Error in evaluating HTML to CSV')
        print(e)

        return {'precision': 0, 'recall': 0, 'f1': 0,"error": str(e)}
