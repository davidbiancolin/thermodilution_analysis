from os.path import join

import analysis.RTLogSegmenting as rtsgm
import analysis.MetricExtraction as metex
from analysis.Plotting import PatientPlotting, AnimalPlotting

from helpers import listdir_clean
import run_params as rp

import matlab.engine
eng = matlab.engine.start_matlab()

# Patient data
if rp.PT_RUN_DATA:
    # Get list of patients
    indiv_patients = listdir_clean(rp.pt_data_folder)
    
    # Final qualification numbers
    final_quals = [878, 850, 860, 865, 870, 847, 851, 865, 871, 858]
    
    # Iterate over patients
    for i, patient in enumerate(indiv_patients):
        # --- PREPROCESSING ---
        # Set up paths to read/save data
        data_path = join(rp.pt_data_folder, patient)
        events_path = join(data_path, 'events.csv')
        save_files = join(rp.pt_saved_files_path, patient)

        # Instantiate class: represents object to concatenate RT logs/segment 
        # data from a given patient
        sgmnt = rtsgm.RTLogSegmenting(data_path, save_files, \
            rp.pt_is_animal_data, rp.pt_desired_cols, rp.pt_tseg_size)
        # Concatenate all patient RT logs
        if rp.PT_CONCAT_LOGS:
            sgmnt.save_concat_rtlogs()
        # Segment patient RT logs into individual CSVs for each thermodilution
        # measurement
        if rp.PT_SEGMENT_LOGS:
            sgmnt.segment_logs()
        
        # --- ANALYSIS ---
        # Set up paths for saving/reading data
        path_to_summary_csv = join(save_files, 'summary.csv')
        segmented_path = join(rp.pt_saved_files_path, patient, \
            'segmented_rtlogs')
        pkl_path = join(rp.pt_saved_files_path, patient, \
            'segment_analysis_pkl')
        
        # Instantiate class: represents all data segments (each 
        # corresponding to a different thermodilution instance) for a given 
        # patient
        ext = metex.MetricExtraction(eng, events_path, segmented_path, \
            pkl_path, path_to_summary_csv, False, \
            ['RTLog Time', ' MC', ' Speed', ' AoP'], final_quals[i])
        # Create pkl files that store information about analyzing each data 
        # segment
        if rp.PT_CREATE_PKLS:
            ext.create_pkls()
        # Manually modify each pkl file to ensure that parameters are correct 
        # for identifying max/min/incisura
        if rp.PT_MANUAL_MODIFY_PKLS:
            ext.manual_modify_pkls()
        # Using the parameters from the pkls, write a summary CSV with all 
        # relevant information
        if rp.PT_WRITE_SUMMARY_CSV:
            ext.write_summary_csv()

    if rp.PT_PLOT_ALL:
        # Create plotting object
        pt_plot = PatientPlotting('processed_data/patient_data')
        # Plot aggregate data (1 point per p-level per patient)
        pt_plot.plot_agg_pts()
        # Plot each thermodilution point (1 point per thermodilution 
        # measurement)
        pt_plot.plot_indiv_pts()
        # Plot each thermodilution point (1 point per thermodilution 
        # measurement) on the same plot
        pt_plot.plot_all_pts()


# Animal Data
if rp.AN_RUN_DATA:
    # Get list of patients
    indiv_animals = listdir_clean(rp.an_data_folder)
    
    # Final qualification numbers
    final_quals = [862, 848, 847, 835, 847]
    for i, animal in enumerate(indiv_animals):
        # --- PREPROCESSING ---
        # Set up paths to read/save data
        data_path = join(rp.an_data_folder, animal)
        events_path = join(data_path, 'events.csv')
        save_files = join(rp.an_saved_files_path, animal)

        # Instantiate class: represents object to concatenate RT logs/segment 
        # data from a given patient
        sgmnt = rtsgm.RTLogSegmenting(data_path, save_files, \
            rp.an_is_animal_data, rp.an_desired_cols, rp.an_tseg_size)
        # Concatenate all patient RT logs
        if rp.AN_CONCAT_LOGS:
            sgmnt.save_concat_rtlogs()
        # Segment patient RT logs into individual CSVs for each thermodilution
        # measurement
        if rp.AN_SEGMENT_LOGS:
            sgmnt.segment_logs()

        # --- ANALYSIS ---
        # Set up paths for saving/reading data
        path_to_summary_csv = join(save_files, 'summary.csv')
        segmented_path = join(rp.an_saved_files_path, animal, \
            'segmented_rtlogs')
        pkl_path = join(rp.an_saved_files_path, animal, \
            'segment_analysis_pkl')

        # Instantiate class: represents all data segments (each 
        # corresponding to a different thermodilution instance) for a given 
        # patient
        ext = metex.MetricExtraction(eng, events_path, segmented_path, \
            pkl_path, path_to_summary_csv, True, \
            ['RTlog_Timestamp', 'MotorCurrent1', 'MotorSpeed1', 'AOP'], \
            final_quals[i])
        # Create pkl files that store information about analyzing each data 
        # segment
        if rp.AN_CREATE_PKLS:
            ext.create_pkls()
        # Manually modify each pkl file to ensure that parameters are correct 
        # for identifying max/min/incisura
        if rp.AN_MANUAL_MODIFY_PKLS:
            ext.manual_modify_pkls()
        # Using the parameters from the pkls, write a summary CSV with all 
        # relevant information
        if rp.AN_WRITE_SUMMARY_CSV:
            ext.write_summary_csv()

    if rp.AN_PLOT_ALL:
        # Create plotting object
        an_plot = AnimalPlotting('processed_data/animal_data')
        # Plot animal data, separating out each physiological state
        an_plot.plot_by_phys_state()
        # Plot animal data in aggregate (1 point per p-level per patient), 
        # separating out each physiological state
        an_plot.plot_agg_norm_state()
        