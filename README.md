# Setup
NOTE: Project used python 3.9.0.
Once repo is cloned from git, in bash shell (to install python modules):
$ cd <project directory>
$ python3.9 -m venv td_env
$ source ./td_env/bin/activate
(td_env) $ python -m pip --version
(td_env) $ pip install --upgrade pip
(td_env) $ pip install -r requirements.txt

To install matlab engine API for python (NOTE: matlab root can be found by 
typing matlabroot in matlab shell in the application)
$ cd "<matlab root>/extern/engines/python"
$ python setup.py install

# Background
This code analyzes the patient SmartPump EFS data and several of the animal datasets. These animal datasets are listed in the Dropbox as:
- 2021 0625 CBSET
- 2021 0707 CBSET
- 2021 0719 CBSET
- 2021 0811 CBSET
- 2021 0812 CBSET

The purpose of this code is to take the raw RT logs from the patient and animal datasets, pull out key parameters (e.g. stroke volume, heart rate, volume of flow through impella, etc.) around the time that each thermodilution measurement occurred, and create a summary CSV summarizing these values. There is also code to make relevant plots from these summary tables.

# Code/Data Structure
Two directories were used to store code/data. 
(1) Firstly, there is the main directory to be saved on the computer you're using. This contains all the python code, along with a few other files necessary for setup/GitHub. The directory is structured:

```
thermodilution_analysis/
├─ analysis/
│  ├─ __init__.py
│  ├─ MetricExtraction.py
│  ├─ Plotting.py
│  ├─ AnalyzeSegment.py
│  ├─ RTLogSegmenting.py
├─ figures/
│  ├─ animals
│  │  ├─ ...
│  ├─ patients
│  │  ├─ ...
├─ processed_data/
│  ├─ animal_data
│  │  ├─ animal_01.csv
│  │  ├─ ...
│  ├─ patient_data
│  │  ├─ patient_01.csv
│  │  ├─ ...
│  ├─ patient_data_prepost
│  │  ├─ patient_01.csv
│  │  ├─ ...
├─ .gitignore
├─ algoFlow.mat
├─ QF_TestQuad.m
├─ README.md
├─ requirements.txt
├─ run_params.py
├─ helpers.py
├─ main.py
```
The analysis/ folder contains the key classes used to do the analysis (MetricExtraction.py, AnalyzeSegment.py, RTLogSegmenting.py) and plotting (Plotting.py). 

The figures/ folder contains all the figures from Plotting.py. They are categorized into patient and animal data, and are subcategorized into other categories. These categories include figures that plot derived values around each thermodilution measurement as a single point, figures that use collective metrics to describe the thermodilution measurements at each P-level (i.e. each point corresponds to some metric derived from 3 thermodilution measurements), different x and y axes, and so on.

The processed_data/ folder contains summary files for the patient (in patient_data/; all 10 patients are shown here) and animal data (in animal_data/; all 5 animals are shown here). Note that the third folder, patient_data_prepost/, contains data from patient_data/, but this time also including the thermodilution timepoints done post-procedure. The post-procedure timepoints were removed manually from the data in the patient_data_prepost/ folder, and the resulting CSVs were put in the patient_data/ folder. In plotting, the data referenced is from patient_data/.

The algoFlow.mat and QF_TestQuad.m files are matlab files from ABiomed that are used to calculate the volume of flow through the impella. run_params.py is used to specify the parameters that are necessary to run the analysis; this keeps all the parameters in one place and makes them easy to change. helpers.py contains useful helper functions that are used throughout the code. main.py is the file that you use to run the code, and calls all the other scripts/classes/methods.

(2) The other directory is where the data is stored. Initially I wanted to access all the data from Dropbox where all the data is stored, but I realized that since some of the files are inconsistently labeled on Dropbox, this made it harder. So, I copied all the files onto an external hard drive (because of the large file sizes) and named them in a way that was consistent so that it would be easier to automate. I also did this so that I could make intermediate CSVs/other datastructures that I could save onto the external hard drive. This speeds up some of the calculations, and allows for the way that the data is being analyzed to be modified without having to perform all the calculations from scratch over and over. This data directory is structured:
```
thermodilution/
├─ animal_data/
│  ├─ animal_01/
│  │  ├─ events.csv
│  │  ├─ RTLogCSV/
│  │  │  ├─ RTLog_20210625T083947_sync_all.csv
│  │  │  ├─ ...
│  ├─ ...
├─ patient_data/
│  ├─ patient_01/
│  │  ├─ events.csv
│  │  ├─ RTLogCSV/
│  │  │  ├─ RTLog_20201105T082317.csv
│  │  │  ├─ ...
│  ├─ ...
├─ processed_animal_data/
│  ├─ animal_01/
│  │  ├─ full_rtlog.csv
│  │  ├─ summary.csv
│  │  ├─ segment_analysis_pkl
│  │  │  ├─ seg_obj00.pkl
│  │  │  ├─ ...
│  │  ├─ segmented_rtlogs
│  │  │  ├─ tslice_00.csv
│  │  │  ├─ ...
│  ├─ ...
├─ processed_patient_data/
│  ├─ patient_01/
│  │  ├─ full_rtlog.csv
│  │  ├─ summary.csv
│  │  ├─ segment_analysis_pkl
│  │  │  ├─ seg_obj00.pkl
│  │  │  ├─ ...
│  │  ├─ segmented_rtlogs
│  │  │  ├─ tslice_00.csv
│  │  │  ├─ ...
│  ├─ ...
```
This data is available on the Dropbox, in Edelman Lab MCS > Data > ThomasU Processed Data > thermodilution.

The data directory, thermodilution/, is divided into folders for the animal and patient data. The animal_data/ and patient_data/ folders contain raw data from the impella RT logs before any data processing is done. There is one folder for each animal/patient. In each folder there is events.csv, which gives the times when each thermodilution measurement was taken. RTLogCSV contains the impella RT log files; there are many files to cover the entire the entire period when the impella was in use.

The processed_animal_data/ and processed_patient_data/ folders contain the processed data. Same as the two other folders in thermodilution/, there is one folder for each animal/patient. In each folder, there is full_rtlog.csv, which has all the RT log files for that patient/animal concatenated. summary.csv is the summary csv, which is the final result from the data processing. segment_analysis_pkl/ contains .pkl files, one for each thermodilution measurement made. These files are binary files that contain instances of the AnalyzeSegment class, and store parameters about how to extract the incisura, minima, and maxima from a segment of data. segmented_rtlogs/ contains "segments" of data from  full_rtlog.csv; each file corresponds to a segment of data from around each thermodilution measurement that is relevant to that measurement, and will be analyzed.

# Data Analysis
Here I will walk through how this code analyzes the patient and animal data. All this computation is done in (1) in the previous section, and the relevant data is read in from/saved to (2) unless otherwise specified.

In order to run calculations, in the bash shell (from the root dierctory of the Github repo):
$ python main.py
The code in this file calls the other classes/methods in analysis/ and uses them to create the summary csv. The parameters that are used in main.py can be modified in run_params; these values are loaded into main.py. 

Firstly, main.py creates an instance of RTLogSegmenting for each patient and animal. RTLogSegmenting has methods that firstly concatenate all the RTLog files into one large file that contains all timepoints. It does this by concatenating through the RTLog files, and while doing so looking for overlap in timepoints between adjacent files. It makes sure that these timepoints are not double counted, and deletes repeat time points. Also, RTLogSegmenting has a method to, given the events.csv file for the patient/animal, separate the full concatenated RTlog into smaller CSVs comprising a few seconds around each thermodilution measurement. It saves these "segments" all as separate files.

Then, main.py creates an instance of MetricExtraction for patient/animal. MetricExtraction controls extracting the relevant metrics for each patient/animal, segment by segment.
Firstly, MetricExtraction creates pkl files for each individual segment. These pkl files contain the relevant information for calculating the summary CSVs, specifically the parameters for the minima and maxima detectors of the aortic pressure, and parameters for the incisura detectors. These pkl files contain instances of AnalyzeSegment, which hold the methods for actually calculating the metrics to be put into the summary CSV, and for calculating each row of the summary CSV (corresponding to a thermodilution measurement). Next, the pkl files are iterated through to manually modify the parameters for each one to ensure proper incisura/min/max detection in the aortic pressure. This is done by inputting values in a certain format in the terminal (instructions are given). Finally, the summary CSV is written and saved.

Finally, Plotting.py contains 2 classes, one for plotting animal data and one for plotting patient data. Both divide up the data based on different criteria, and generally plot metrics like standard deviation of stroke volume or heart rate on the x axis, and difference between the thermodilution values and calculated values on the y axis. The full results can be seen in the figures folder.
