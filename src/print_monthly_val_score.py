import click
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import DT_M, JOIN_KEY_COLUMN, read_target


def calc_auc(df):
    return roc_auc_score(df.isFraud.values, df.Prediction.values)


@click.command()
@click.option('--conf_name', type=str, default='lgbm_0000')
def main(conf_name) -> None:
    df = pd.read_csv(f'../data/output/{conf_name}/oof.csv').rename({'isFraud': 'Prediction'}, axis=1)
    df['isFraud'] = read_target()
    df['DT_M'] = DT_M().create_feature()[0]['DT_M']

    print(f'whole data : {calc_auc(df)}')
    for month in df['DT_M'].unique():
        temp_df = df[df['DT_M'] == month].reset_index()
        print(f'month_{month}_{calc_auc(temp_df)}')


if __name__ == "__main__":
    main()
