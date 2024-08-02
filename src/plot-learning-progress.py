import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

prog_df = pd.read_csv(sys.argv[1])
cols = ['val_loss', *prog_df.columns[prog_df.columns.str.contains('val_auc')].to_list()]
prog_df = prog_df.set_index('epoch')[cols].dropna()
sns.lineplot(prog_df)
plt.title('Are we learning yet?')
plt.suptitle(sys.argv[1].split('/')[-5:-1])
plt.savefig(sys.argv[2])
plt.close()

# invoke as python ~/schan/quantum/src/plot-learning-progress.py $(readlink -f metrics.csv )
