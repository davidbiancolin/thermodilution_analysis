from os.path import join

# ---------------- RUN SPECIFICATIONS ----------------
# --- Patient data ---
PT_RUN_DATA = False
# - Preprocessing -
PT_CONCAT_LOGS = False
PT_SEGMENT_LOGS = False
# - Analysis -
PT_CREATE_PKLS = False
PT_MANUAL_MODIFY_PKLS = False
PT_WRITE_SUMMARY_CSV = False
# - Plotting -
PT_PLOT_ALL = False


# --- Animal data ---
AN_RUN_DATA = True
# - Preprocessing -
AN_CONCAT_LOGS = False
AN_SEGMENT_LOGS = False
# - Analysis -
AN_CREATE_PKLS = False
AN_MANUAL_MODIFY_PKLS = False
AN_WRITE_SUMMARY_CSV = True
# - Plotting -
AN_PLOT_ALL = False

# ---------------- ARGUMENTS & PATH SPECIFICATIONS ----------------
main_data_folder = '/Volumes/thomas_5tb_drive/PhD Files/coding/' + \
    'edelman_lab/thermodilution/'
# --- Patient data ---
# PREPROCESSING
pt_data_folder = join(main_data_folder, 'patient_data')
pt_saved_files_path = join(main_data_folder, 'processed_patient_data')

pt_is_animal_data = False
pt_desired_cols = ['RTLog Time', ' MC', ' Speed', ' AoP', ' dP']
pt_tseg_size = 13

# --- Animal data ---
# PREPROCESSING
an_data_folder = join(main_data_folder, 'animal_data')
an_saved_files_path = join(main_data_folder, 'processed_animal_data')

an_is_animal_data = True
an_desired_cols = ['RTlog_Timestamp', 'MotorSpeed1', 'MotorCurrent1', 'AOP', 'LVP', \
    'LVV']
an_tseg_size = 21
