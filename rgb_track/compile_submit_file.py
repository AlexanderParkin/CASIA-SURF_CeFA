import pandas as pd

if __name__ == '__main__':
    submit_file_df = pd.DataFrame()
    for idx in range(1, 4):
        df = pd.read_csv(f'experiment_tests/protocol_4_{idx}/rgb_track/exp1_protocol4_{idx}/TestFileLogger/output_4.csv')
        submit_file_df = submit_file_df.append(df[['video_id', 'output_score']], ignore_index=True)
    submit_file_df.to_csv('./submit_file.txt', index=False, header=False, sep=' ')