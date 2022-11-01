import pandas as pd
from os.path import join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import listdir_clean

class PatientPlotting:
    """
    Class to plot all patient data in a variety of formats
    """
    def __init__(self, folder_path, show=False):
        """
        Constructor for PatientPlotting

        Args:
            folder_path (str): path to folder of processed CSVs to plot
            show (bool, optional): _description_. Whether to show plots, or 
                just save them without showing. Defaults to False.
        """
        self.file_labels = ['Patient 1', 'Patient 2', 'Patient 3', \
            'Patient 4', 'Patient 5', 'Patient 6', 'Patient 7', 'Patient 8', \
            'Patient 10']
        self.csv_dfs = self.get_all_csvs(folder_path)
        self.lvefs = [.40,.25,.45,.45,.60,.40,.45,.55,.45]
        self.show_plots = show


    def get_all_csvs(self, folder_path):
        """
        Get a list of dataframes corresponding with the 

        Args:
            folder_path (str): path to folders containing all processed csvs

        Returns:
            list(DataFrame): list of dataframes containing 
        """
        # Get all file names for processed data files
        fnames = listdir_clean(folder_path)
        
        # Iterate through processed data files, load them to dataframes, and
        # add them to a python list
        all_dfs = []
        for f in fnames:
            full_name = join(folder_path, f)
            all_dfs.append(pd.read_csv(full_name))
        return all_dfs


    def plot_agg_pts(self):
        """
        Plots aggregate patient data (1 point per p-level per patient, 
        represents y-axis as range of differences)
        """
        # P values to iterate through
        p_vals = ['P8', 'P6', 'P4', 'P2']
        for p in p_vals:
            # Initialize arrays
            mean_std_hr = np.zeros(len(self.csv_dfs))
            mean_std_pp = np.zeros(len(self.csv_dfs))
            mean_std_imp_fv = np.zeros(len(self.csv_dfs))

            calc_diff = np.zeros(len(self.csv_dfs))
            pinging_diff = np.zeros(len(self.csv_dfs))

            # Iterate through csvs with processed data
            for i, df in enumerate(self.csv_dfs):
                # Dataframe sliced based on a certain p-value
                df_sl = df.loc[df['P_val'] == p]

                # Get normalized HR, PP, impella FV for the given p-value
                mean_std_hr[i] = np.mean(df_sl['std_hr'] / df_sl['avg_hr'])
                mean_std_pp[i] = np.mean(df_sl['std_pp'] / df_sl['avg_pp'])
                mean_std_imp_fv[i] = np.mean(
                    df_sl['std_imp_fv'] / df_sl['avg_imp_fv'])

                # Differences between calculated/pinging CO and thermodilution
                # CO
                calc_diff[i] = np.ptp(df_sl['total_co'] - df_sl['CO'])
                pinging_diff[i] = np.ptp(df_sl['pinging_co'] - df_sl['CO'])
            
            # Set colormap
            cm = plt.cm.get_cmap('Reds')
            
            # Plot normalized mean std. HR versus thermodilution CO - 
            # calculated CO
            fig = plt.figure()
            sc = plt.scatter(mean_std_hr, calc_diff, c=self.lvefs, cmap=cm)
            plt.colorbar(sc)
            plt.xlabel('Normalized Mean Std. Heart Rate (bpm)')
            plt.ylabel('Range of Cardiac Output Difference (TD CO - Calc. CO)')
            plt.title(p)
            if self.show_plots: plt.show()
            fig.savefig(f'figures/patients/aggregate/hr_std/calc/{p}')

            # Plot normalized mean std. PP versus thermodilution CO - 
            # calculated CO
            fig = plt.figure()
            sc = plt.scatter(mean_std_pp, calc_diff, c=self.lvefs, cmap=cm)
            plt.colorbar(sc)
            plt.xlabel('Normalized Mean Std. Pulse Pressure (mmHg)')
            plt.ylabel('Range of Cardiac Output Difference (TD CO - Calc. CO)')
            plt.title(p)
            if self.show_plots: plt.show()
            fig.savefig(f'figures/patients/aggregate/pp_std/calc/{p}')

            # Plot normalized mean std. impella FV versus thermodilution CO - 
            # calculated CO
            fig = plt.figure()
            sc = plt.scatter(mean_std_imp_fv, calc_diff, c=self.lvefs, cmap=cm)
            plt.colorbar(sc)
            plt.xlabel('Normalized Mean Std. Impella Flow Volume (mL)')
            plt.ylabel('Range of Cardiac Output Difference (TD CO - Calc. CO)')
            plt.title(p)
            if self.show_plots: plt.show()
            fig.savefig(f'figures/patients/aggregate/imp_fv_std/calc/{p}')

            # Plot normalized mean std. HR versus thermodilution CO - 
            # pinging CO
            fig = plt.figure()
            sc = plt.scatter(mean_std_hr, pinging_diff, c=self.lvefs, cmap=cm)
            plt.colorbar(sc)
            plt.xlabel('Normalized Mean Std. Heart Rate (bpm)')
            plt.ylabel(
                'Range of Cardiac Output Difference (TD CO - Pinging CO)')
            plt.title(p)
            if self.show_plots: plt.show()
            fig.savefig(f'figures/patients/aggregate/hr_std/pinging/{p}')

            # Plot normalized mean std. PP versus thermodilution CO - 
            # pinging CO
            fig = plt.figure()
            sc = plt.scatter(mean_std_pp, pinging_diff, c=self.lvefs, cmap=cm)
            plt.colorbar(sc)
            plt.xlabel('Normalized Mean Std. Pulse Pressure (mmHg)')
            plt.ylabel(
                'Range of Cardiac Output Difference (TD CO - Pinging CO)')
            plt.title(p)
            if self.show_plots: plt.show()
            fig.savefig(f'figures/patients/aggregate/pp_std/pinging/{p}')

            # Plot normalized mean std. impella FV versus thermodilution CO - 
            # pinging CO
            fig = plt.figure()
            sc = plt.scatter(mean_std_imp_fv, pinging_diff, c=self.lvefs, cmap=cm)
            plt.colorbar(sc)
            plt.xlabel('Normalized Mean Std. Impella Flow Volume (mL)')
            plt.ylabel(
                'Range of Cardiac Output Difference (TD CO - Pinging CO)')
            plt.title(p)
            if self.show_plots: plt.show()
            fig.savefig(f'figures/patients/aggregate/imp_fv_std/pinging/{p}')

    
    def plot_indiv_pts(self):
        """
        Plots individual thermodilution points, one plot per patient
        """
        # Iterate through patients
        for i, (df, lbl) in enumerate(zip(self.csv_dfs, self.file_labels)):
            # Patient identifier string
            str_pt_num = 'patient_' + format(i+1, '02d')
            
            # Create new columns in dataframe for metrics
            df['calc_diff'] = df['CO'] - df['total_co']
            df['pinging_diff'] = df['CO'] - df['pinging_co']
            df['norm_std_hr'] = df['std_hr'] / df['avg_hr']
            df['norm_std_pp'] = df['std_pp'] / df['avg_pp']
            df['norm_std_imp_fv'] = df['std_imp_fv'] / df['avg_imp_fv']
            
            # Plot norm. std. HR versus CO Difference (thermodilution - 
            # calculated)
            fig = sns.lmplot(x="norm_std_hr", y="calc_diff", \
                hue="P_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Heart Rate (bpm)')
            plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
            plt.title(lbl)
            if self.show_plots: plt.show()
            fig.savefig(
                f'figures/patients/indiv_patients/hr_std/calc/{str_pt_num}')

            # Plot norm. std. PP versus CO Difference (thermodilution - 
            # calculated)
            fig = sns.lmplot(x="norm_std_pp", y="calc_diff", \
                hue="P_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Pulse Pressure (mmHg)')
            plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
            plt.title(lbl)
            if self.show_plots: plt.show()
            fig.savefig(
                f'figures/patients/indiv_patients/pp_std/calc/{str_pt_num}')

            # Plot norm. std. impella FV versus CO Difference (thermodilution - 
            # calculated)
            fig = sns.lmplot(x="norm_std_imp_fv", y="calc_diff", \
                hue="P_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Impella Flow Volume (mL)')
            plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
            plt.title(lbl)
            if self.show_plots: plt.show()
            fig.savefig(
                f'figures/patients/indiv_patients/imp_fv_std/calc/{str_pt_num}')

            # Plot norm. std. HR versus CO Difference (thermodilution - 
            # pinging)
            fig = sns.lmplot(x="norm_std_hr", y="pinging_diff", \
                hue="P_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Heart Rate (bpm)')
            plt.ylabel('Cardiac Output Difference (TD CO - Pinging CO)')
            plt.title(lbl)
            if self.show_plots: plt.show()
            fig.savefig(
                f'figures/patients/indiv_patients/hr_std/pinging/{str_pt_num}')

            # Plot norm. std. PP versus CO Difference (thermodilution - 
            # pinging)
            fig = sns.lmplot(x="norm_std_pp", y="pinging_diff", \
                hue="P_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Pulse Pressure (mmHg)')
            plt.ylabel('Cardiac Output Difference (TD CO - Pinging CO)')
            plt.title(lbl)
            if self.show_plots: plt.show()
            fig.savefig(
                f'figures/patients/indiv_patients/pp_std/pinging/{str_pt_num}')

            # Plot norm. std. impella FV versus CO Difference (thermodilution - 
            # pinging)
            fig = sns.lmplot(x="norm_std_imp_fv", y="pinging_diff", \
                hue="P_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Impella Flow Volume (mL)')
            plt.ylabel('Cardiac Output Difference (TD CO - Pinging CO)')
            plt.title(lbl)
            if self.show_plots: plt.show()
            fig.savefig(
                'figures/patients/indiv_patients/imp_fv_std/pinging/' + \
                f'{str_pt_num}')


    def plot_all_pts(self):
        """
        Plot individual thermodilution points on the same plot (for all 
        patients)
        """
        # Concatenate all dataframes for all patients
        df = pd.concat(self.csv_dfs, axis=0).reset_index(drop=True)

        # Create new columns in dataframe for different metrics 
        df['calc_diff'] = df['CO'] - df['total_co']
        df['pinging_diff'] = df['CO'] - df['pinging_co']
        df['norm_std_hr'] = df['std_hr'] / df['avg_hr']
        df['norm_std_pp'] = df['std_pp'] / df['avg_pp']
        df['norm_std_imp_fv'] = df['std_imp_fv'] / df['avg_imp_fv']
        
        # Plot norm. std. HR versus CO difference between thermodilution CO 
        # and calculated CO
        fig = sns.lmplot(x="norm_std_hr", y="calc_diff", \
                hue="P_val", data=df, fit_reg=False)
        plt.xlabel('Normalized Std. Heart Rate (bpm)')
        plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
        if self.show_plots: plt.show()
        fig.savefig(
            f'figures/patients/all_patients/hr_std_calc.png')

        # Plot norm. std. PP versus CO difference between thermodilution CO 
        # and calculated CO
        fig = sns.lmplot(x="norm_std_pp", y="calc_diff", \
            hue="P_val", data=df, fit_reg=False)
        plt.xlabel('Normalized Std. Pulse Pressure (mmHg)')
        plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
        if self.show_plots: plt.show()
        fig.savefig(
            f'figures/patients/all_patients/pp_std_calc.png')

        # Plot norm. std. impella FV versus CO difference between 
        # thermodilution CO and calculated CO
        fig = sns.lmplot(x="norm_std_imp_fv", y="calc_diff", \
            hue="P_val", data=df, fit_reg=False)
        plt.xlabel('Normalized Std. Impella Flow Volume (mL)')
        plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
        if self.show_plots: plt.show()
        fig.savefig(
            f'figures/patients/all_patients/imp_fv_std_calc.png')
        
        # Plot norm. std. HR versus CO difference between thermodilution CO 
        # and pinging CO
        fig = sns.lmplot(x="norm_std_hr", y="pinging_diff", \
            hue="P_val", data=df, fit_reg=False)
        plt.xlabel('Normalized Std. Heart Rate (bpm)')
        plt.ylabel('Cardiac Output Difference (TD CO - Pinging CO)')
        if self.show_plots: plt.show()
        fig.savefig(
            f'figures/patients/all_patients/hr_std_pinging.png')

        # Plot norm. std. PP versus CO difference between thermodilution CO 
        # and pinging CO
        fig = sns.lmplot(x="norm_std_pp", y="pinging_diff", \
            hue="P_val", data=df, fit_reg=False)
        plt.xlabel('Normalized Std. Pulse Pressure (mmHg)')
        plt.ylabel('Cardiac Output Difference (TD CO - Pinging CO)')
        if self.show_plots: plt.show()
        fig.savefig(
            f'figures/patients/all_patients/pp_std_pinging.png')

        # Plot norm. std. impella FV versus CO difference between 
        # thermodilution CO and calculated CO
        fig = sns.lmplot(x="norm_std_imp_fv", y="pinging_diff", \
            hue="P_val", data=df, fit_reg=False)
        plt.xlabel('Normalized Std. Impella Flow Volume (mL)')
        plt.ylabel('Cardiac Output Difference (TD CO - Pinging CO)')
        if self.show_plots: plt.show()
        fig.savefig(
            'figures/patients/all_patients/imp_fv_std_pinging.png')


class AnimalPlotting:
    """
    Class to plot all animal data in a variety of formats
    """
    def __init__(self, folder_path, show=False):
        """
        Constructor for AnimalPlotting

        Args:
            folder_path (str): path to folder of processed CSVs to plot
            show (bool, optional): _description_. Whether to show plots, or 
                just save them without showing. Defaults to False.
        """
        self.file_labels = ['Animal Study: 06/25/2021', \
            'Animal Study: 07/07/2021 CBSET', \
            'Animal Study: 07/19/2021 CBSET', \
            'Animal Study: 08/11/2021 CBSET', \
            'Animal Study: 08/12/2021 CBSET']
        self.csv_dfs = self.get_all_csvs(folder_path)
        self.show_plots = show


    def get_all_csvs(self, folder_path):
        # Get all file names for processed data files
        fnames = listdir_clean(folder_path)

        # Iterate through processed data files, load them to dataframes, and
        # add them to a python list
        all_dfs = []
        for f in fnames:
            full_name = join(folder_path, f)
            all_dfs.append(pd.read_csv(full_name))
        return all_dfs


    def plot_by_phys_state(self):
        """
        Plot individual plots for each physiological state. Each datapoint 
        represents an individual thermodilution measurement
        """
        # Concatenate all dataframes for each animal
        df_concat = pd.concat(self.csv_dfs, axis=0).reset_index(drop=True)

        # Make new columns in dataframe for different metrics
        df_concat['calc_diff'] = df_concat['CO'] - df_concat['total_co']
        df_concat['norm_std_hr'] = df_concat['std_hr'] / df_concat['avg_hr']
        df_concat['norm_std_pp'] = df_concat['std_pp'] / df_concat['avg_pp']
        df_concat['norm_std_imp_fv'] = \
            df_concat['std_imp_fv'] / df_concat['avg_imp_fv']

        # Iterate through all physiological states
        phys_states = ['baseline', 'imp_normal', 'CS']
        for st in phys_states:
            # Slice dataframe to just contain rows with a certain physiological
            # state
            df = df_concat.loc[df_concat['label'] == st]
            
            # Plot norm. std. HR versus CO difference (thermodilution - 
            # calculated)
            fig = sns.lmplot(x="norm_std_hr", y="calc_diff", \
                hue="p_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Heart Rate (bpm)')
            plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
            if self.show_plots: plt.show()
            fig.savefig(
                join('figures/animals/by_physio_state', st, 'std_hr.png'))

            # Plot norm. std. PP versus CO difference (thermodilution - 
            # calculated)
            fig = sns.lmplot(x="norm_std_pp", y="calc_diff", \
                hue="p_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Pulse Pressure (mmHg)')
            plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
            if self.show_plots: plt.show()
            fig.savefig(
                join('figures/animals/by_physio_state', st, 'std_pp.png'))

            # Plot norm. std. impella FV versus CO difference (thermodilution - 
            # calculated)
            fig = sns.lmplot(x="norm_std_imp_fv", y="calc_diff", \
                hue="p_val", data=df, fit_reg=False)
            plt.xlabel('Normalized Std. Impella Flow Volume (mL)')
            plt.ylabel('Cardiac Output Difference (TD CO - Calc. CO)')
            if self.show_plots: plt.show()
            fig.savefig(
                join('figures/animals/by_physio_state', st, 'std_imp_fv.png'))


    def plot_agg_norm_state(self):
        """
        Plot aggregate data (1 datapoint per p-level) against range of CO 
        differences. One plot per physiological state
        """
        # Iterate through physiological states
        phys_states = ['imp_normal', 'CS']
        for st in phys_states:
            # Set up dataframe for aggregate data
            agg_df = pd.DataFrame()
            
            # Iterate through each animal's processed data CSV
            for df in self.csv_dfs:
                # Slice out data for a certain physiological state from the 
                # animal's DF
                df = df.loc[df['label'] == st]

                # Create new columns in dataframe for metrics
                df['calc_diff'] = df['CO'] - df['total_co']
                df['norm_std_hr'] = df['std_hr'] / df['avg_hr']
                df['norm_std_pp'] = df['std_pp'] / df['avg_pp']
                df['norm_std_imp_fv'] = df['std_imp_fv'] / df['avg_imp_fv']
                
                # Iterate through each p-value in sliced dataframe
                p_vals = ['P8', 'P6', 'P4', 'P2']
                for p in p_vals:
                    # Slice out thermodiliton measurements at a certain 
                    # p-value from the dataframe
                    df_sl = df.loc[df['p_val'] == p]
                    if not df_sl.empty:
                        # Create a row for the dataframe, and append it to the 
                        # full dataframe
                        add_to_df = pd.DataFrame(
                            data={'p_val':[p], \
                                'norm_std_hr':[np.mean(df_sl['norm_std_hr'])], \
                                'norm_std_pp':[np.mean(df_sl['norm_std_pp'])], \
                                'norm_std_imp_fv':[np.mean(df_sl['norm_std_imp_fv'])], \
                                'calc_diff':[np.ptp(df_sl['calc_diff'])]})
                        agg_df = pd.concat([agg_df, add_to_df], axis=0)
            
            agg_df = agg_df.reset_index(drop=True)

            # Plot norm. std. HR versus range of CO differences 
            # (thermodilution CO - calculated CO)
            fig = sns.lmplot(x="norm_std_hr", y="calc_diff", \
                hue="p_val", data=agg_df, fit_reg=False)
            plt.xlabel('Normalized Std. Heart Rate (bpm)')
            plt.ylabel('Range of Cardiac Output Difference (TD CO - Calc. CO)')
            if self.show_plots: plt.show()
            fig.savefig(
                join('figures/animals/aggregate', st, 'std_hr.png'))

            # Plot norm. std. PP versus range of CO differences 
            # (thermodilution CO - calculated CO)
            fig = sns.lmplot(x="norm_std_pp", y="calc_diff", \
                hue="p_val", data=agg_df, fit_reg=False)
            plt.xlabel('Normalized Std. Pulse Pressure (mmHg)')
            plt.ylabel('Range of Cardiac Output Difference (TD CO - Calc. CO)')
            if self.show_plots: plt.show()
            fig.savefig(
                join('figures/animals/aggregate', st, 'std_pp.png'))

            # Plot norm. std. impella FV versus range of CO differences 
            # (thermodilution CO - calculated CO)
            fig = sns.lmplot(x="norm_std_imp_fv", y="calc_diff", \
                hue="p_val", data=agg_df, fit_reg=False)
            plt.xlabel('Normalized Std. Impella Flow Volume (mL)')
            plt.ylabel('Range of Cardiac Output Difference (TD CO - Calc. CO)')
            if self.show_plots: plt.show()
            fig.savefig(
                join('figures/animals/aggregate', st, 'std_imp_fv.png'))