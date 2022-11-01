from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from helpers import timedelta_to_seconds, matlab_to_numpy, numpy_to_matlab, load_table_dt

class AnalyzeSegment:
    """
    Datastructure for the data around a single thermodilution measurement. 
    Stores parameters used to detect local maxima/minima and incisura. Has 
    methods to store this instance of the class as a .pkl file, so that its 
    parameters can be referenced later. Has methods to modify these parameters 
    to more accurately detect min/max/incisura. Using these parameters, has 
    methods to calculate average and std. HR, average and std. PP, average 
    and std. impella FV, and calculated CO, and save these values in a 
    dataframe to be written into the summary csv.
    """
    def __init__(self, eng, path_to_seg_csv, desired_cols, final_qual, \
        hr_std_samp_size=3):
        """
        Constructor for AnalyzeSegment

        Args:
            eng (matlab.engine): instance of matlab engine
            path_to_seg_csv (str): path to csv containing data for this 
                thermodilution segment
            desired_cols (list(str)): desired columns to be used. should be in 
                this order: [<time column name>, <current column name>, 
                <speed column name>, <aortic pressure column name>]
            final_qual (int): Final qualification number for this patient/animal
        """
        self.matlab_engine = eng

        self.path_to_seg_csv = path_to_seg_csv
        
        # Split the segment CSV for this thermodilution value into columns to 
        # be used later 
        self.time, self.mc, self.spd, self.aop = \
            self._split_seg_csv(desired_cols)

        self.hr_std_samp_size = hr_std_samp_size

        # -- Set default parameters for max/min detector --
        # minimum distance (in units of indices in the self.aop array) in 
        # which maxima in aortic pressure can be detected
        self.max_dist = 150
        
        # minimum prominence allowed for aortic pressure maxima
        self.max_prom_min = 4

        # minimum distance (in units of indices in the self.aop array) in 
        # which minima in aortic pressure can be detected
        self.min_dist = 150

        # minimum prominence allowed for aortic pressure minima
        self.min_prom_min = 4

        # number between 0 and 1, indicating the minimum of where the 
        # incisura algorithm should search for the incisura point within 
        # each beat. Each beat is defined to start at the detected maximum 
        # and end at the next minimum, so a value of 0.5 here would mean 
        # that the algorithm should only search in the end 1/2 of the 
        # beat to find the incisura index
        self.incis_search_min = 0.1

        # number between 0 and 1, indicating the maximum of where the 
        # incisura algorithm should search for the incisura point within 
        # each beat. Each beat is defined to start at the detected maximum 
        # and end at the next minimum, so a value of 0.5 here would mean 
        # that the algorithm should only search in the beginning 1/2 of the 
        # beat to find the incisura index
        self.incis_search_max = 0.8

        # Final qualification number for this patient/animal
        self.final_qual = final_qual


    def _split_seg_csv(self, desired_cols):
        """
        Split CSV into columns. Each column is a different metric (time, 
        aortic pressure, etc.) used for the calculations in this class.
        Returns tuple of np.ndarray.

        Args:
            desired_cols (list(str)): desired columns to be used. should be in 
                this order: [<time column name>, <current column name>, 
                <speed column name>, <aortic pressure column name>] 

        Returns:
            tuple(np.ndarray): tuple of columns representing time, motor 
            current, motor speed, aortic pressure respectively
        """
        # Read in full table for this thermodilution segment
        full_rtlog = load_table_dt(self.path_to_seg_csv, desired_cols[0], \
            desired_cols)

        # Assign each column
        time_col = full_rtlog[desired_cols[0]].to_numpy()
        current_col = full_rtlog[desired_cols[1]].to_numpy()
        speed_col = full_rtlog[desired_cols[2]].to_numpy()
        aop_col = full_rtlog[desired_cols[3]].to_numpy()
        
        return time_col, current_col, speed_col, aop_col


    def calc_max_min_inds(self):
        """
        Using the parameters for the max/min detectors in this class, find 
        max/min values in aortic pressure

        Returns:
            tuple(np.ndarray): ndarrays for the indices of the maximum and 
            minimum values in this thermodilution segment
        """
        # Use max/min detectors
        pks_max, _ = find_peaks(self.aop, distance=self.max_dist, \
            prominence=(self.max_prom_min, None))
        pks_min, _ = find_peaks(-self.aop, distance=self.min_dist, \
            prominence=(self.min_prom_min, None))
        
        # Validate these maximum and minimum indices
        val_max, val_min = self._validate_pks(pks_max, pks_min)
        val_max = val_max.astype(int)
        val_min = val_min.astype(int)
        return val_max, val_min


    def _validate_pks(self, pks_max, pks_min):
        """
        Validates the peak values in pks_max and pks_min. Ensures that each
        maximum is followed by one corresponding minimum, eliminating repeat
        maxima or minima within the same beat by iterating through the indices
        of the maxima and minima (pks_max and pks_min) in the aortic pressure

        Args:
            pks_max (np.ndarray): indices (in self.aop) of the maxima of the 
                aortic pressure
            pks_min (np.ndarray): indices (in self.aop) of the minima of the 
                aortic pressure

        Returns:
            tuple(np.ndarray): indices of the maxima and minima respectively 
                in self.aop
        """
        # Create arrays for the validated maxima and minima
        val_pks_max = np.empty(min(len(pks_max), len(pks_min)))
        val_pks_max[:] = np.nan
        val_pks_min = np.empty(min(len(pks_max), len(pks_min)))
        val_pks_min[:] = np.nan

        # Iterate through all max and min indices
        i_max = 0
        i_min = 0
        iter_counter = 0
        while True:
            # Get rid of all minimum indices before the first maximum index
            while pks_max[i_max] > pks_min[i_min] and i_min < len(pks_min):
                i_min += 1

            # Indices of next minimum and next maximum
            next_i_max = i_max + 1
            next_i_min = i_min + 1
            
            # If at end of pks_max or pks_min array, end
            if next_i_max >= len(pks_max) or next_i_min >= len(pks_min):
                break
           
           # Pick the earliest maximum index that comes after the minimum of 
           # the current beat (i.e. the beat with max index i_max and min 
           # index i_min)
            while pks_max[next_i_max] < pks_min[i_min] and \
                next_i_max < len(pks_max):
                next_i_max += 1
                if next_i_max >= len(pks_max):
                    break
            
            # Pick the latest minimum index out of the minimum indices before 
            # the next maximum 
            if next_i_max < len(pks_max):
                while pks_min[next_i_min] < pks_max[next_i_max] and \
                    next_i_min < len(pks_min):
                    next_i_min += 1
                    if next_i_min >= len(pks_min):
                        break
            
            # Add this validated maximum and minimum to arrays
            val_pks_max[iter_counter] = pks_max[i_max]
            val_pks_min[iter_counter] = pks_min[next_i_min-1]
            iter_counter += 1

            # Reset max and min indices for next iteration
            i_max = next_i_max; i_min = next_i_min;
            if i_max >= len(pks_max) or i_min >= len(pks_min):
                break
        
        # Strip away any NaNs in the validated arrays of max and min indices,
        # and return them
        val_pks_max_clean = val_pks_max[np.logical_not(np.isnan(val_pks_max))]
        val_pks_min_clean = val_pks_min[np.logical_not(np.isnan(val_pks_min))]
        return val_pks_max_clean, val_pks_min_clean


    def calc_incisura(self, pks_max, pks_min):
        """
        Given validated indices of maxima and minima within the aortic pressure
        data, calculate the indices of the incisura within each beat (i.e. 
        within the range of the maximum and minimum index of each beat)

        Args:
            pks_max (np.ndarray): indices (in self.aop) of the maxima of the 
                aortic pressure
            pks_min (np.ndarray): indices (in self.aop) of the minima of the 
                aortic pressure

        Returns:
            tuple(np.ndarray): tuple of the incisura indices, maxima of the 
            search ranges within each beat, and the minima of the search 
            ranges within each beat
        """
        # Array for incisura indices
        incis_idxs = np.zeros(len(pks_max))
        
        # Arrays for the maximum and minimum of the search range of indices 
        # (to search for the incisura) within each beat
        search_rng_maxs = np.zeros(len(pks_max))
        search_rng_mins = np.zeros(len(pks_max))

        # Iterate through all maxima and minima
        for i, (max_idx, min_idx) in enumerate(zip(pks_max, pks_min)):
            # Lower index to search for the incisura within this beat
            start_idx = max_idx + int(np.rint((min_idx - max_idx) * \
                self.incis_search_min))
            
            # Upper index to search for the incisura within this beat
            end_idx = max_idx + int(np.rint((min_idx - max_idx) * \
                self.incis_search_max))
            
            # Slice out of aortic pressure
            search_slice = self.aop[start_idx:end_idx]

            # Max of 2nd derivative in slice
            deriv_max = np.argmax(np.gradient(np.gradient(search_slice)))
            
            # Add incisura index, search range max and min to arrays
            incis_idxs[i] = start_idx + deriv_max
            search_rng_maxs[i] = end_idx
            search_rng_mins[i] = start_idx
        return incis_idxs.astype(int), search_rng_maxs, search_rng_mins

    
    def manual_modify_params(self):
        """
        Function for manually modifying parameters for this instance of 
        AnalyzeSegment. Plots a graph of the aortic pressure tracing for this
        data segment, and shows the locations of the detected maxima, minima, 
        incisura, and incisura search ranges. Shows the user the current set of 
        parameters, and prompts the user to either use new parameters (input 
        new parameters with commas in between in the same order they are 
        displayed) or end execution when the parameters are satisfactory (type 
        "end"). These new parameters are later saved to a pkl file when 
        manually modifying parameters in bulk in an instance of MetricExtraction

        Returns:
            AnalyzeSemgnet: Returns an instance of the class, this time with 
            updated parameters, ready to be saved to a pkl file.
        """
        
        # Iterate until the parameters are deemed satisfactory
        optimized = False
        while not optimized:
            # Using current parameter set, get the indices of the maxima, 
            # minima, incisura, and min and max of incisura search range
            pks_max, pks_min = self.calc_max_min_inds()
            incis_idxs, search_rng_maxs, search_rng_mins = self.calc_incisura(pks_max, pks_min)

            # Plot graph of aortic pressure tracing (vertical lines show 
            # locations of important indices)
            print('Color code: red = maxes, blue = mins, black = incisura, ' + \
                'green = max/min of incisura search range')
            plt.plot([i for i in range(len(self.aop))], self.aop)
            for pk_mx in pks_max:
                plt.axvline(pk_mx, color='r', linewidth=1)
            for pk_mn in pks_min:
                plt.axvline(pk_mn, color='b', linewidth=1)
            for idx in incis_idxs:
                plt.axvline(idx, color='k', linewidth=1)
            for rng_max in search_rng_maxs:
                plt.axvline(rng_max, color='g', linewidth=0.75)
            for rng_min in search_rng_mins:
                plt.axvline(rng_min, color='g', linewidth=0.75)
            plt.show()
            
            # Print out old parameter set
            print('-- Input new params for max/min/incisura detection --')
            print('Old params:')
            max_dist = self.get_max_dist()
            print(f'    max_dist = {max_dist}')
            max_prom_min = self.get_max_prom_min()
            print(f'    max_prom_min = {max_prom_min}')
            min_dist = self.get_min_dist()
            print(f'    min_dist = {min_dist}')
            min_prom_min = self.get_min_prom_min()
            print(f'    min_prom_min = {min_prom_min}')
            incis_search_min = self.get_incis_search_min()
            print(f'    incis_search_max = {incis_search_min}')
            incis_search_max = self.get_incis_search_max()
            print(f'    incis_search_max = {incis_search_max}')

            print('Input format: "max_dist","max_prom_min","min_dist",' + \
                '"min_prom_min","incis_search_min","incis_search_max"')
            print('Type "end" to move to next thermodilution segment')

            # Prompt for new parameter set
            new_params = [max_dist, max_prom_min, min_dist, min_prom_min, \
                incis_search_min, incis_search_max]
            bad_input = True
            while bad_input:
                try:
                    # Prompt for input
                    input_str = input('New params: ')
                    
                    # If "end", then stop execution of the loop
                    if input_str == 'end':
                        optimized = True
                        bad_input = False
                    else:
                        # Split input into list of parameters
                        new_params = \
                            [float(i) for i in str.split(input_str, ',')]
                except:
                    print('Invalid input. Try again.')
                else:
                    # Must return 6 inputs, 1 for each parameter
                    if len(new_params) == 6:
                        bad_input = False
            
            # Set parameters to new inputs
            self.set_max_dist(new_params[0])
            self.set_max_prom_min(new_params[1])
            self.set_min_dist(new_params[2])
            self.set_min_prom_min(new_params[3])
            self.set_incis_search_min(new_params[4])
            self.set_incis_search_max(new_params[5])
        return self


    def calc_df_row(self):
        """
        Calculate a row of the dataframe that will eventually be the summary 
        CSV with important derived metrics from this thermodilution data 
        segment.

        Returns:
            pd.DataFrame: row of the summary csv DataFrame
        """
        # Calculate indices of maxima and minima in aortic pressure, and 
        # convert these indices to times and aortic pressure values
        pks_max, pks_min = self.calc_max_min_inds()
        pks_max_time = self.time[pks_max]
        pks_max_dep_var = self.aop[pks_max]
        pks_min_dep_var = self.aop[pks_min]
        
        # -- Compute average HR --
        # Calculate total time to complete the heartbeat sample in this segment
        avg_hr_time = timedelta_to_seconds(pks_max_time[-1] - pks_max_time[0])
        # Divide number of peaks by total time taken
        avg_hr = len(pks_max) / (avg_hr_time / 60)

        # -- Compute std HR --
        # Reshape array of heart rate max times to more easily group them to 
        # calculate standard deviations: force the array into a number of 
        # columns equal to the sample size for calculating std, and take the 
        # std of each row of the matrix; this is the list of running std's for 
        # this segment, so you can take the average to get the relevant std. 
        # HR value
        num_pks = len(pks_max)
        grouped_hr_times = np.reshape(
            pks_max_time[:num_pks - num_pks % self.hr_std_samp_size], \
            (int(num_pks / self.hr_std_samp_size), self.hr_std_samp_size))
        
        # Get durations for each row of the matrix above (to compute the heart 
        # rate for each row), convert to seconds, and take the std
        seg_durations = grouped_hr_times[:,self.hr_std_samp_size-1] - \
            grouped_hr_times[:,0]
        seg_duration_mins = timedelta_to_seconds(seg_durations) / 60
        std_hr = np.std(self.hr_std_samp_size / seg_duration_mins)

        # -- Compute average PP --
        # Compute average pulse pressure by subtracting values at the minimum 
        # of the aortic pressure from the values at the maximum, and take the 
        # mean
        pp_vals = pks_max_dep_var - pks_min_dep_var
        avg_pp = np.mean(pp_vals)

        # -- Compute std PP --
        # Compute std pulse pressure by taking the standard deviation of the 
        # values from before
        std_pp = np.std(pp_vals)

        # -- Compute average impella flow volume and std impella flow volume --
        # Find incisura indices and calculate impella flow
        incis_idxs, _, _ = self.calc_incisura(pks_max, pks_min)
        imp_fv_arr = self._calc_impella_flow(pks_max, pks_min, incis_idxs)
        
        # Eliminate negative values of impella flow, as this most likely 
        # indicates a motor speed of 0
        if np.mean(imp_fv_arr) > 0:
            avg_imp_fv = np.mean(imp_fv_arr)
            std_imp_fv = np.std(imp_fv_arr)
        else:
            avg_imp_fv = 0
            std_imp_fv = 0

        # -- Compute total cardiac output --
        total_co = avg_hr * (avg_pp + avg_imp_fv) / 1000

        # Infer speed setting of impella. Average the speed over the segment, 
        # and choose the P-value closest to that value
        spd_dict = {0:'P0', 23000:'P1', 31000:'P2', 33000:'P3', 35000:'P4', \
            37000:'P5', 39000:'P6', 42000:'P7', 44000:'P8'}
        spds = np.array([i for i in spd_dict.keys()])
        min_idx = np.argmin(np.abs(np.mean(self.spd * 10) - spds))
        infer_spd = spd_dict[spds[min_idx]]

        # Making dataframe row from derived values, to be added to summary csv
        data_dict = {'avg_hr':[avg_hr], 'std_hr':[std_hr], 'avg_pp':[avg_pp], \
            'std_pp':[std_pp], 'avg_imp_fv':[avg_imp_fv], \
            'std_imp_fv':[std_imp_fv], 'total_co':[total_co], \
            'infer_spd':[infer_spd]}
        return pd.DataFrame(data=data_dict)
        

    def _calc_impella_flow(self, pks_max, pks_min, incis_idxs):
        """
        Using maximum and minimum indices in the aortic pressure, as well as 
        the incisura indices, calculate

        Args:
            pks_max (np.ndarray): indices (in self.aop) of the maxima of the 
                aortic pressure
            pks_min (np.ndarray): indices (in self.aop) of the minima of the 
                aortic pressure
            incis_idxs (np.ndarray): indices (in self.aop) of the incisura in 
                the aortic pressure

        Returns:
            np.ndarray: array of impella flow values (each element in array is 
            flow within 1 beat of the sample)
        """
        # Iterate through minimum and incisura indices in self.aop
        impella_flows = np.zeros(len(pks_min))
        for i, (min_idx, incis_idx) in enumerate(zip(pks_min, incis_idxs)):
            # Set up an array to pass into QF_TestQuad (function for 
            # calculating the flow volume in the impella)
            arr_for_testquad = np.zeros((min_idx - incis_idx, 3))
            arr_for_testquad[:,0] = self.mc[incis_idx:min_idx] / 1e3
            arr_for_testquad[:,1] = self.spd[incis_idx:min_idx] * 10
            arr_for_testquad[:,2] = np.ones((min_idx - incis_idx)) * \
                self.final_qual
            
            # Convert this array to a matlab array
            matlab_arr = numpy_to_matlab(arr_for_testquad)
            
            # Pass into matlab QF_TestQuad function
            flow_raw = matlab_to_numpy(
                self.matlab_engine.QF_TestQuad(matlab_arr))

            # Multiply each impella flow value from above with the time over 
            # which that flow occurred
            time_segments = timedelta_to_seconds(
                np.diff(self.time[incis_idx:min_idx+1])) / 60
            impella_flows[i] = sum(flow_raw.flatten() * time_segments)
        return impella_flows
            

    def get_max_dist(self):
        """
        Getter for self.max_dist

        Returns:
            float: minimum distance (in units of indices in the self.aop 
            array) in which maxima in aortic pressure can be detected
        """
        return self.max_dist

    
    def get_min_dist(self):
        """
        Getter for self.min_dist

        Returns:
            float: minimum distance (in units of indices in the self.aop 
            array) in which minima in aortic pressure can be detected
        """
        return self.min_dist


    def get_max_prom_min(self):
        """
        Getter for self.max_prom_min

        Returns:
            float: minimum prominence allowed for aortic pressure maxima
        """
        return self.max_prom_min


    def get_min_prom_min(self):
        """
        Getter for self.min_prom_min

        Returns:
            float: minimum prominence allowed for aortic pressure minima
        """
        return self.min_prom_min


    def get_incis_search_max(self):
        """
        Getter for self.incis_search_max

        Returns:
            float: number between 0 and 1, indicating the maximum of where the 
            incisura algorithm should search for the incisura point within 
            each beat. Each beat is defined to start at the detected maximum 
            and end at the next minimum, so a value of 0.5 here would mean 
            that the algorithm should only search in the beginning 1/2 of the 
            beat to find the incisura index
        """
        return self.incis_search_max


    def get_incis_search_min(self):
        """
        Getter for self.incis_search_min

        Returns:
            float: number between 0 and 1, indicating the minimum of where the 
            incisura algorithm should search for the incisura point within 
            each beat. Each beat is defined to start at the detected maximum 
            and end at the next minimum, so a value of 0.5 here would mean 
            that the algorithm should only search in the end 1/2 of the 
            beat to find the incisura index
        """
        return self.incis_search_min


    def set_max_dist(self, max_dist):
        """
        Setter for self.max_dist

        Args:
            max_dist (float): minimum distance (in units of indices in the 
            self.aop array) in which maxima in aortic pressure can be detected
        """
        self.max_dist = max_dist


    def set_min_dist(self, min_dist):
        """
        Setter for self.min_dist

        Args:
            min_dist (float): minimum distance (in units of indices in the 
            self.aop array) in which minima in aortic pressure can be detected
        """
        self.min_dist = min_dist


    def set_max_prom_min(self, max_prom_min):
        """
        Setter for self.max_prom_min

        Args:
            max_prom_min (float): minimum prominence allowed for aortic 
            pressure maxima
        """
        self.max_prom_min = max_prom_min


    def set_min_prom_min(self, min_prom_min):
        """
        Setter for self.min_prom_min

        Args:
            min_prom_min (float): minimum prominence allowed for aortic 
            pressure minima
        """
        self.min_prom_min = min_prom_min


    def set_incis_search_max(self, incis_search_max):
        """
        Setter for self.incis_search_max

        Args:
            incis_search_max (float): number between 0 and 1, indicating the 
            maximum of where the incisura algorithm should search for the 
            incisura point within each beat. Each beat is defined to start at 
            the detected maximum and end at the next minimum, so a value of 
            0.5 here would mean that the algorithm should only search in the 
            beginning 1/2 of the beat to find the incisura index
        """
        self.incis_search_max = incis_search_max


    def set_incis_search_min(self, incis_search_min):
        """
        Setter for self.incis_search_min

        Args:
            incis_search_min (float): number between 0 and 1, indicating the 
            minimum of where the incisura algorithm should search for the 
            incisura point within each beat. Each beat is defined to start at 
            the detected maximum and end at the next minimum, so a value of 
            0.5 here would mean that the algorithm should only search in the 
            end 1/2 of the beat to find the incisura index
        """
        self.incis_search_min = incis_search_min


    def set_matlab_engine(self, matlab_engine):
        """
        Setter for self.matlab_engine

        Args:
            matlab_engine (matlab.engine): matlab engine object for running 
            matlab scripts
        """
        self.matlab_engine = matlab_engine


    def __getstate__(self):
        """
        Returns a dictionary representation of the class's fields. Used when
        generating .pkl files from instances of this class.

        Returns:
            dict: dictionary of the fields of this instance of the class
        """
        attrs_dict = self.__dict__

        # Delete the matlab engine from the class's fields, since it doesn't 
        # have a __getstate__ method and causes problems otherwise
        if 'matlab_engine' in attrs_dict.keys():
            del attrs_dict['matlab_engine']
        return attrs_dict