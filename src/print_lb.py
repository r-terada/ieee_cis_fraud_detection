import os

def main():
    print('print lb score')
    tmp = os.popen(f'kaggle competitions submissions -c ieee-fraud-detection -v | head -n 8').read()
    col, *values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for val in values:
        for i,j in zip(col.split(','), val.split(',')):
            message += f'{i}: {j}\n'
    print(message.rstrip())

if __name__ == "__main__":
    main()
