import os
import time
import glob
import click
import subprocess


@click.command()
@click.option('--conf_name', type=str, default='lgbm_0000')
def main(conf_name) -> None:
    out_dir = os.path.join('../data/output/', conf_name)
    sub_path = os.path.join(out_dir, 'submission.csv')
    cv_score = ""
    for fname in glob.glob(os.path.join(out_dir, "score_*")):
        cv_score = fname.split('_')[-1]

    message = f'{conf_name}:cv_{cv_score}' if cv_score else conf_name

    print(message)

    submit_cmd_list = f'''
        kaggle
        competitions
        submit -c ieee-fraud-detection
        -f {sub_path}
        -m {message}
    '''.split()

    print(f'submit: {submit_cmd_list}')
    subprocess.run(submit_cmd_list)


if __name__ == '__main__':
    main()
