'''
https://www.kaggle.com/paulorzp/gmean-of-low-correlation-lb-0-952x
'''

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

from scipy.stats import describe
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    LABELS = ["isFraud"]
    all_files = glob.glob("../data/external/gmean_of_low_correlation/*.csv")
    all_files.remove('../data/external/gmean_of_low_correlation/submission-.9485.csv')
    my_subs = [
        # my subs
        "../data/output/lgbm_045/submission.csv",
    ]
    scores = []
    for i in range(len(all_files)):
        scores.append(float('.' + all_files[i].split(".")[3]))
    # my subs
    for s, path in zip([0.9499], my_subs):
        scores.append(s)
        all_files.append(path)
    
    scores = np.array(scores)

    top = scores.argsort()[::-1]
    for i, f in enumerate(top):
        print(i, scores[f], all_files[f])
    
    outs = [pd.read_csv(all_files[f], engine='python').set_index('TransactionID') for f in top]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols

    print(concat_sub)

    # check correlation
    corr = concat_sub.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

    # Draw the heatmap with the mask and correct aspect ratio
    _ = sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
                    annot=True,fmt='.4f', cbar_kws={"shrink":.2})
    
    plt.savefig('subs_corr.png')

    mean_corr = corr.mean()
    mean_corr = mean_corr.sort_values(ascending=True)
    mean_corr = mean_corr[:5]
    print(mean_corr)

    m_gmean1 = 0
    for n in mean_corr.index:
        m_gmean1 += np.log(concat_sub[n])
    m_gmean1 = np.exp(m_gmean1/len(mean_corr))

    rank = np.tril(corr.values,-1)
    rank[rank<0.92] = 1
    m = (rank>0).sum() - (rank>0.97).sum()
    m_gmean2, s = 0, 0
    for n in range(m):
        mx = np.unravel_index(rank.argmin(), rank.shape)
        w = (m-n)/m
        m_gmean2 += w*(np.log(concat_sub.iloc[:,mx[0]])+np.log(concat_sub.iloc[:,mx[1]]))/2
        s += w
        rank[mx] = 1
    m_gmean2 = np.exp(m_gmean2/s)

    top_mean = 0
    s = 0
    for n in [0, 1, 4, 25]:
        top_mean += concat_sub.iloc[:,n]*scores[top[n]]
        s += scores[top[n]]
    top_mean /= s

    m_gmean = np.exp(0.3*np.log(m_gmean1) + 0.15*np.log(m_gmean2) + 0.55*np.log(top_mean))
    print(describe(m_gmean))

    concat_sub['isFraud'] = m_gmean
    if not os.path.exists('../data/output/stack_gmean_005'):
        os.makedirs('../data/output/stack_gmean_005')
    concat_sub[['isFraud']].to_csv('../data/output/stack_gmean_005/submission.csv')


if __name__ == "__main__":
    main()
