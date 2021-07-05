import pandas as pd
from collections import defaultdict
import argparse
import sys

def eprint(args):
    sys.stderr.write(str(args) + "\n")
# Parse arguments
parser = argparse.ArgumentParser(
    description='')

parser.add_argument('--files', nargs='*', help="files") 

args = parser.parse_args()
mydict = defaultdict()
for f in args.files:
    df_history = pd.read_csv(f)
    df_history.sort_values('val_dice_coef', inplace=True)
    mydict[f.split('/')[-1]] = df_history.iloc[0]['val_dice_coef']

df_out = pd.DataFrame.from_dict(mydict,orient='index')
df_out.to_csv('history_merged.csv')