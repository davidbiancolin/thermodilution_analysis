import pandas as pd
import os
import numpy as np
import pickle

from analysis.AnalyzeSegment import AnalyzeSegment
from helpers import listdir_clean, load_table_dt

class MetricExtraction:
    """
    From segmented RT log files, performs peak detection on AoP data to
    determine pulse pressure/heart rate, impella flow, and LVEF (if applicable)
    """

    def __init__(self, eng, path_to_events_csv, path_to_seg_csvs, \
        path_to_pkls, path_to_summary_csv, is_animal_data, desired_cols, \
        final_qual):
        """
        Constructor for MetricExtraction

        Args:
            eng (matlab.engine): instance of matlab engine
            path_to_events_csv (str): path to CSV withe events data, defining 
                when the thermodilution measurements took place
            path_to_seg_csvs (str): path of the folder containing the data for 
                each data segment corresponding to each 
            path_to_summary_csv (str): path where the summary csv will be saved
            is_animal_data (bool): defines if filepaths refer to animal or 
                patient data
            desired_cols (list(str)): list of desired columns from segment 
                data to be considered. should be in this order: 
                [<time column name>, <current column name>, 
                <speed column name>, <aortic pressure column name>]
        """
        self.matlab_engine = eng
        self.path_to_events_csv = path_to_events_csv
        self.path_to_seg_csvs = path_to_seg_csvs
        self.path_to_pkls = path_to_pkls
        self.path_to_summary_csv = path_to_summary_csv
        self.is_animal_data = is_animal_data
        self.desired_cols = desired_cols
        self.final_qual = final_qual
        self.pkl_fname = 'seg_obj'


    def create_pkls(self):
        """
        Creates .pkl files based on the segmented data corresponding with each 
        thermodilution measurement. pkl files contain parameters for detecting 
        minima/maxima/incisura; these parameters can be modified by modifying 
        the contents of these files
        """
        seg_paths = listdir_clean(self.path_to_seg_csvs)
        for i, pth in enumerate(seg_paths):
            full_seg_pth = os.path.join(self.path_to_seg_csvs, pth)
            seg = AnalyzeSegment(self.matlab_engine, full_seg_pth, \
                self.desired_cols, self.final_qual, hr_std_samp_size=3)
            str_td_num = format(i, '02d')
            pkl_name = os.path.join(
                self.path_to_pkls, self.pkl_fname + str_td_num + '.pkl')
            with open(pkl_name, 'wb') as handle:
                pickle.dump(seg, handle)


    def manual_modify_pkls(self):
        """
        Using the manual_modify_params function in AnalyzeSegment, iterate 
        through each pkl file and manually modify the parameters for detecting 
        minima/maxima/incisura
        """
        seg_obj_paths = listdir_clean(self.path_to_pkls)
        for pth in seg_obj_paths:
            full_pth = os.path.join(self.path_to_pkls, pth)
            segment_obj = self.load_pkl(full_pth)
            new_dict = segment_obj.manual_modify_params()
            with open(full_pth, 'wb') as handle:
                pickle.dump(new_dict, handle)


    def write_summary_csv(self):
        """
        Creates the summary CSV containing relevant calculated values for 
        each thermodilution measurement
        """
        full_df = pd.DataFrame()
        seg_obj_paths = listdir_clean(self.path_to_pkls)
        for i, pth in enumerate(seg_obj_paths):
            segment_obj = self.load_pkl(pth)
            segment_obj.set_matlab_engine(self.matlab_engine)
            df_row = segment_obj.calc_df_row()
            full_df = pd.concat([full_df, df_row], axis=0)

        events_table = load_table_dt(
            self.path_to_events_csv, 'DateTime', ['DateTime', 'CO'])
        full_df = full_df.reset_index(drop=True)
        full_df = pd.concat([events_table['DateTime'], full_df], axis=1)
        full_df = pd.concat([full_df, events_table['CO']], axis=1)
        full_df.to_csv(self.path_to_summary_csv, index=False)


    def load_pkl(self, pth):
        """
        Loads a pkl file and returns it

        Args:
            pth (str): path to pkl file

        Returns:
            AnalyzeSegment: an AnalyzeSegment object that contains parameter 
            information from the pkl file path passed into this function
        """
        pkl_path = os.path.join(self.path_to_pkls, pth)
        with open(pkl_path, 'rb') as handle:
            segment_obj = pickle.load(handle)
        return segment_obj