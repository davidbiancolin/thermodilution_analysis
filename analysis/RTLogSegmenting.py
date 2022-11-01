import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from helpers import listdir_clean, load_table_dt, inrange

class RTLogSegmenting:
    """
    Takes a folder containing thermodilution and RT log data for one patient/
    animal, concatenates all RT log csvs, and segments the RT log around 
    each thermodilution measurement. Pickles data segments and saves them.
    """

    def __init__(self, data_path, saved_files_path, is_animal_data, 
        desired_cols, tseg_size, suppress_text=False):
        """
        Constructor for RTLogSegmenting.

        Args:
            data_path (str): path to folder containing event and rtlog data
            saved_files_path (str): path to folder where processed data is saved
            is_animal_data (bool): indicates if data is animal or patient data
            desired_cols (list[str]): column headers in rtlogs to be extracted 
                in processed data. first column assumed to be time values (in 
                the format mm/dd/yyyy hh:mm:ss.000).
            tseg_size (int): time length (seconds) of segments segmented from 
                full RT logs
        """
        # TODO: THIS IS AWFUL. FIX IT EXPLICITLY
        self.events_path, self.rtlogs_path = self._get_data_subpaths(data_path)
        self.saved_files_path = saved_files_path
        self.full_rtlog_path = \
            os.path.join(self.saved_files_path, 'full_rtlog.csv')
        self.is_animal_data = is_animal_data
        self.desired_cols = desired_cols
        self.tseg_size = tseg_size 
        self.suppress_text = suppress_text
        
        # Size of chunks that are read from full RT log csvs
        self.chunk_size = 1e4


    def _get_data_subpaths(self, data_path):
        contents = listdir_clean(data_path)
        events_path = os.path.join(data_path, contents[0])
        rtlogs_path = os.path.join(data_path, contents[1])
        return (events_path, rtlogs_path)


    def save_concat_rtlogs(self):
        csv_list = listdir_clean(self.rtlogs_path)

        if self.is_animal_data:
            self._concat_animal_data(csv_list)
        else:
            self._concat_patient_data(csv_list)


    def _concat_animal_data(self, csv_list):
        if not self.suppress_text:
            print('Concatenating and saving animal data...')
        for i, f in enumerate(csv_list):
            if not self.suppress_text:
                print(f'Concat files: {i+1} of {len(csv_list)}')
            df_path = os.path.join(self.rtlogs_path, f)
            df = load_table_dt(df_path, self.desired_cols[0], \
                usecols=self.desired_cols)
            df.to_csv(self.full_rtlog_path, mode='a', index=False,
                header=not os.path.exists(self.full_rtlog_path))


    def _concat_patient_data(self, csv_list):
        if not self.suppress_text:
            print('Concatenating and saving patient data...')
        df_list = []
        for f in csv_list:
            df_path = os.path.join(self.rtlogs_path, f)
            df = load_table_dt(df_path, self.desired_cols[0], \
                usecols=self.desired_cols)
            df_list.append(df)
        
        full_rtlog_raw = pd.concat(df_list)
        full_rtlog = self._elim_overlap(full_rtlog_raw)
        full_rtlog.to_csv(self.full_rtlog_path, index=False)

    
    def _elim_overlap(self, full_rtlog_raw):
        if not self.suppress_text:
            print('Eliminating overlaps in patient data...')
        num_raw_pts = len(full_rtlog_raw)
        rep_time_mask = np.ones(num_raw_pts, dtype=bool)
        time_col = full_rtlog_raw[self.desired_cols[0]]
        
        if not self.suppress_text:
            iterator = tqdm(range(1, num_raw_pts))
        else:
            iterator = range(1, num_raw_pts)
        for t_i in iterator:
            if time_col.iloc[t_i] < time_col.iloc[t_i-1]:
                logical_rep_vals = np.zeros(num_raw_pts, dtype=bool);
                logical_rep_vals[:t_i-1] = \
                    (time_col[:t_i-1] >= time_col.iloc[t_i]).to_numpy()
                rep_time_mask[logical_rep_vals] = False

        return full_rtlog_raw.loc[rep_time_mask]


    def segment_logs(self):
        if not self.suppress_text:
            print('Segmenting concatenated RT logs...')
        path_to_save = os.path.join(self.saved_files_path, 'segmented_rtlogs')

        start_times = load_table_dt(self.events_path, 'DateTime', \
            ['DateTime', 'CO'])['DateTime']
        end_times = start_times + pd.DateOffset(seconds=self.tseg_size)

        continue_segs_arr = np.zeros(start_times.shape, dtype=bool)
        already_saved = np.zeros(start_times.shape, dtype=bool)
        with pd.read_csv(self.full_rtlog_path, chunksize=self.chunk_size) \
            as reader:
            for chunk in reader:
                time_col = self.desired_cols[0]
                chunk[time_col] = pd.to_datetime(chunk[time_col])
                c_start = chunk[time_col].iloc[0]
                c_end = chunk[time_col].iloc[-1]

                if not self.suppress_text:
                    print(f'Times: {c_start} to {c_end}')
                
                for i, _ in enumerate(already_saved):
                    td_start = start_times[i]
                    td_end = end_times[i]

                    contains_start = inrange(td_start, c_start, c_end)
                    contains_end = inrange(td_end, c_start, c_end)

                    if not contains_start and contains_end:
                        if not self.suppress_text:
                            print('--Segment contains end--')
                            print(f'          Saving td pt {i} of ' + \
                                f'{len(already_saved)}')
                        final_slice = \
                            chunk[chunk[time_col] < td_end]
                        sliced_arr = arr_slice_segment.append(final_slice)
                        continue_segs_arr[i] = False

                        str_td_num = format(i, '02d')
                        file_save_path = os.path.join(path_to_save, f'tslice_{str_td_num}.csv')
                        sliced_arr.to_csv(file_save_path, index=False)
                        already_saved[i] = True
                        if np.all(already_saved):
                            break
                    
                    if contains_start and contains_end:
                        if not self.suppress_text:
                            print('--Segment contains start & end--')
                            print(f'          Saving td pt {i} of ' + \
                                f'{len(already_saved)}')
                        logic_arr = np.logical_and(chunk[time_col] < td_end, chunk[time_col] > td_start)
                        sliced_arr = chunk[logic_arr]

                        str_td_num = format(i, '02d')
                        file_save_path = os.path.join(path_to_save, f'tslice_{str_td_num}.csv')
                        sliced_arr.to_csv(file_save_path, index=False)
                        already_saved[i] = True
                        if np.all(already_saved):
                            break

                    if contains_start and not contains_end:
                        if not self.suppress_text:
                            print('--Segment contains start--')
                        arr_slice_segment = chunk[chunk[time_col] > td_start]
                        continue_segs_arr[i] = True

                    if not contains_start and not contains_end and \
                        continue_segs_arr[i]:
                        arr_slice_segment = arr_slice_segment.append(chunk)






