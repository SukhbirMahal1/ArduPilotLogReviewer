import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
from scipy import signal

from .ArduPilotFilterReviewer import ArduPilotFilterReviewer
class ArduPilotLogReviewer:
    def __init__(self, 
                 filedate:str, 
                 filepath:str, 
                 MOTOR_RCOU_CH:list, 
                 auto_detect_flight:bool=True, 
                 T_MIN:int=None, 
                 T_MAX:int=None, 
                 ESC_CONT_A:int=None, 
                 ESC_BURST_A:int=None, 
                 save_plots:bool=True, 
                 show_plots:bool=True,
                 verbose:bool=True):
        
        self.filedate = filedate
        self.filepath = filepath
        self.MOTOR_RCOU_CH = MOTOR_RCOU_CH
        self.auto_detect_flight = auto_detect_flight
        self.T_MIN = T_MIN
        self.T_MAX = T_MAX
        self.ESC_CONT_A = ESC_CONT_A
        self.ESC_BURST_A = ESC_BURST_A
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.verbose = verbose

        # constants
        self.DES_HOVER_PWM = (1600, 1700)

        # directories
        os.makedirs(f'csvs/{self.filedate}', exist_ok=True)
        os.makedirs(f'plots/{self.filedate}', exist_ok=True)
        os.makedirs(f'summaries/{self.filedate}', exist_ok=True)

        # parse
        self.mavlog = MavLog(filepath=filepath)
        print(f'Parsing {filepath}...')
        self.mavlog.parse()
        self.parm = pd.DataFrame(self.mavlog.get('PARM').fields)

        # detect flight window
        if auto_detect_flight:
            [self.T_MIN, self.T_MAX] = self._detect_flight_window()
        else:
            if T_MIN is None or T_MAX is None:
                raise ValueError('T_MIN and T_MAX must be provided when auto_detect_flight=False')
            self.T_MIN = T_MIN
            self.T_MAX = T_MAX
            if self.verbose:
                print(f'Flight Window: {self.T_MIN:.1f}s - {self.T_MAX:.1f}s (Duration: {self.T_MAX - self.T_MIN:.1f}s)')
        
        self.rcou = self._get_msg('RCOU')

    def plot_att(self):
        msg = 'ATT'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        self._save_csv(df, msg)

        error_roll = df['Roll'] - df['DesRoll']
        error_pitch = df['Pitch'] - df['DesPitch']
        error_yaw = df['Yaw'] - df['DesYaw']

        rmse_roll = np.sqrt(np.mean(error_roll ** 2))
        rmse_pitch = np.sqrt(np.mean(error_pitch ** 2))
        rmse_yaw = np.sqrt(np.mean(error_yaw ** 2))

        plt.figure(figsize=(21, 2))

        plt.subplot(1, 3, 1)
        plt.plot(df['TimeUS'] / 1e6, df['DesRoll'], label=msg+'.DesRoll')
        plt.plot(df['TimeUS'] / 1e6, df['Roll'], label=msg+'.Roll')
        plt.xlabel('Time [s]')
        plt.ylabel('Roll [$^\circ$]')
        plt.title(f'RMSE: {rmse_roll:.1f}$^\circ$')
        plt.grid(1)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(df['TimeUS'] / 1e6, df['DesPitch'], label=msg+'.DesPitch')
        plt.plot(df['TimeUS'] / 1e6, df['Pitch'], label=msg+'.Pitch')
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch [$^\circ$]')
        plt.title(f'RMSE: {rmse_pitch:.1f}$^\circ$') 
        plt.grid(1)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(df['TimeUS'] / 1e6, df['DesYaw'], label=msg+'.DesYaw')
        plt.plot(df['TimeUS'] / 1e6, df['Yaw'], label=msg+'.Yaw')
        plt.xlabel('Time [s]')
        plt.ylabel('Yaw [$^\circ$]')
        plt.title(f'RMSE: {rmse_yaw:.1f}$^\circ$') 
        plt.grid(1)
        plt.legend()

        self._save_plot(msg)
    
    def plot_vibes(self):
        msg = 'VIBE'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        imu_groups = df.groupby('IMU')

        plt.figure(figsize=(7 * df['IMU'].nunique(), 2))
        
        for i, idx in imu_groups:
            idx = idx.reset_index(drop=True)
            self._save_csv(idx, msg+f'[{i}]')
            plt.subplot(1, df['IMU'].nunique(), i + 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['VibeX'], label=f'IMU[{i}].VibeX')
            plt.plot(idx['TimeUS'] / 1e6, idx['VibeY'], label=f'IMU[{i}].VibeY')
            plt.plot(idx['TimeUS'] / 1e6, idx['VibeZ'], label=f'IMU[{i}].VibeZ')
            plt.axhline(30, color='red', label='Minimum Safe Vibrations')
            plt.xlabel('Time [s]')
            plt.ylabel('Vibrations [m/s$^2$]')
            plt.grid(1)
            plt.legend()

        self._save_plot(msg)

    def plot_rcou(self):
        msg = 'RCOU'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        self._save_csv(df, msg)

        plt.figure(figsize=(7, 2))

        for i, idx in enumerate(self.MOTOR_RCOU_CH):
            plt.plot(df['TimeUS'] / 1e6, df[f'C{idx}'], label=msg+f'.C{idx}')

        plt.axhspan(*self.DES_HOVER_PWM, color='gray', alpha=0.3, label='Des. Hover PWM')
        
        plt.xlabel('Time [s]')
        plt.ylabel('PWM [$\mu$s]')
        plt.grid(1)
        plt.legend()
        self._save_plot(msg)
    
    def plot_esc(self):
        msg = 'ESC'
        print(f'Plotting '+msg+'...')
        df = self._get_msg(msg=msg)
        esc_groups = df.groupby('Instance')

        plt.figure(figsize=(7, 2))

        for i, idx in esc_groups:
            idx = idx.reset_index(drop=True)
            self._save_csv(idx, f'ESC[{i}].RPM')

            plt.plot(idx['TimeUS'] / 1e6, idx['RPM'], label=f'ESC[{i}].RPM')
            
        plt.xlabel('Time [s]')
        plt.ylabel('RPM')
        plt.grid(1)
        plt.legend()
            
        self._save_plot(msg)

    def plot_bat(self):
        msg = 'BAT'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        bat_groups = df.groupby('Inst')

        try:
            bat_volt_min = self.parm.query("Name == 'Q_M_BAT_VOLT_MIN'")['Value'].iloc[0]
            bat_volt_min_label = 'Q_M_BAT_VOLT_MIN'
        except:
            bat_volt_min = self.parm.query("Name == 'MOT_BAT_VOLT_MIN'")['Value'].iloc[0]
            bat_volt_min_label = 'MOT_BAT_VOLT_MIN'

        plt.figure(figsize=(14, 2))

        for i, idx in bat_groups:
            idx = idx.reset_index(drop=True)
            self._save_csv(idx, msg+f'[{i}]')

            plt.subplot(1, 2, 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['VoltR'], label=f'BAT[{i}].VoltR')
            plt.axhline(bat_volt_min, color='red', label=bat_volt_min_label)
            plt.xlabel('Time [s]')
            plt.ylabel('Voltage [V]')
            plt.grid(1)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(idx['TimeUS'] / 1e6, idx['Curr'], label=f'BAT[{i}].Curr')
            if self.ESC_CONT_A is not None:
                plt.axhline(self.ESC_CONT_A, color='red', label='ESC Continuous Current Limit')
            if self.ESC_BURST_A is not None:
                plt.axhline(self.ESC_BURST_A, color='green', label='ESC Burst Current Limit')
            plt.xlabel('Time [s]')
            plt.ylabel('Current [A]')
            plt.grid(1)
            plt.legend()
            
        self._save_plot(msg)

    def plot_imus(self):
        msg = 'IMU'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        imu_groups = df.groupby('I')

        plt.figure(figsize=(7 * df['I'].nunique(), 4))

        for i, idx in imu_groups:
            idx = idx.reset_index(drop=True)
            self._save_csv(idx, msg+f'[{i}]')

            plt.subplot(2, df['I'].nunique(), i + 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['GyrX'], label=f'IMU[{i}].GyrX')
            plt.plot(idx['TimeUS'] / 1e6, idx['GyrY'], label=f'IMU[{i}].GyrY')
            plt.plot(idx['TimeUS'] / 1e6, idx['GyrZ'], label=f'IMU[{i}].GyrZ')
            plt.xlabel('Time [s]')
            plt.ylabel('[rad/s]')
            plt.grid(1)
            plt.legend()

            plt.subplot(2, df['I'].nunique(), i + df['I'].nunique() + 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['AccX'], label=f'IMU[{i}].AccX')
            plt.plot(idx['TimeUS'] / 1e6, idx['AccY'], label=f'IMU[{i}].AccY')
            plt.plot(idx['TimeUS'] / 1e6, idx['AccZ'], label=f'IMU[{i}].AccZ')
            plt.xlabel('Time [s]')
            plt.ylabel('[m/s]')
            plt.grid(1)
            plt.legend()
            
        self._save_plot(msg)

    def plot_powr(self):
        msg = 'POWR'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        self._save_csv(df, msg)

        plt.figure(figsize=(7, 2))
        plt.plot(df['TimeUS'] / 1e6, df['Vcc'], label=msg+'.Vcc')
        plt.plot(df['TimeUS'] / 1e6, df['VServo'], label=msg+'.VServo')
        plt.axhline(5, color='red', label='Minimum FCU Safe Voltage')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.grid(1)
        plt.legend()
        self._save_plot(msg)

    def plot_compass_interference(self):
        msg = 'MAG'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        mag_groups = df.groupby('I')

        rcou_ = self.rcou[['TimeUS', f'C{self.MOTOR_RCOU_CH[0]}']].copy()
        pwm_min = self.parm.loc[self.parm['Name'] == f'SERVO{self.MOTOR_RCOU_CH[0]}_MIN', 'Value'].iloc[0]
        pwm_max = self.parm.loc[self.parm['Name'] == f'SERVO{self.MOTOR_RCOU_CH[0]}_MAX', 'Value'].iloc[0]

        plt.figure(figsize=(7 * df['I'].nunique(), 2))

        for i, idx in mag_groups:
            idx = idx.reset_index(drop=True)

            mag_field = np.sqrt(idx['MagX']**2 + idx['MagY']**2 + idx['MagZ']**2)

            throttle_interp = np.interp(idx['TimeUS'], rcou_['TimeUS'], (rcou_[f'C{self.MOTOR_RCOU_CH[0]}'] - pwm_min) / (pwm_max - pwm_min) * 100)

            baseline_window = throttle_interp > 20
            if not np.any(baseline_window):
                continue

            baseline_mag = mag_field[baseline_window].mean()
            baseline_times = idx['TimeUS'][baseline_window]

            baseline_start = baseline_times.iloc[0]
            baseline_end = baseline_times.iloc[-1]

            mag_interference = ((mag_field - baseline_mag) / baseline_mag) * 100
            mag_interference_max = mag_interference.max()
            
            if mag_interference_max < 30:
                assessment = 'Acceptable'
                color = 'green'
            elif mag_interference_max < 60:
                assessment = 'Gray Zone (30-60%)'
                color = 'orange'
            else:
                assessment = 'Bad (>60%)'
                color = 'red'

            mag_df = pd.DataFrame({
                'TimeUS': idx['TimeUS'],
                'MagField': mag_field,
                'Throttle': throttle_interp,
            })

            self._save_csv(mag_df, msg+f'[{i}]')

            plt.subplot(1, df['I'].nunique(), i + 1)
            plt.plot(idx['TimeUS'] / 1e6, mag_field, label=f'MAG[{i}] Field Strength [mGauss]')
            plt.plot(idx['TimeUS'] / 1e6, throttle_interp, label='Throttle [%]')
            plt.axvspan(baseline_start /1e6, baseline_end /1e6, alpha=0.3, color='gray', label=f'Mag{[i]} Interference Calculation Window')
            plt.xlabel('Time [s]')
            plt.ylabel('')
            plt.grid(1)
            plt.legend()
            plt.title(f'MAG[{i}] Interference {assessment}: {mag_interference_max:.1f}%', color=color)
        
        self._save_plot(msg)

    def plot_gps(self):
        msg = 'GPS'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        gps_groups = df.groupby('I')

        plt.figure(figsize=(7 * df['I'].nunique(), 4))

        for i, idx in gps_groups:
            idx = idx.reset_index(drop=True)
            self._save_csv(idx, msg+f'[{i}]')

            plt.subplot(2, df['I'].nunique(), i + 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['HDop'], label=f'GPS[{i}].HDop]')
            plt.axhline(2, color='red', label='Not Good')
            plt.axhline(1.5,color='green', label='Very Good')
            plt.xlabel('Time [s]')
            plt.ylabel('HDOP [m]')
            plt.grid(1)
            plt.legend()

            plt.subplot(2, df['I'].nunique(), i + df['I'].nunique() + 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['NSats'], label=f'GPS[{i}].NSats]')
            plt.axhline(12, color='red', label='Bad')
            plt.xlabel('Time [s]')
            plt.ylabel('Number of Satellites')
            plt.grid(1)
            plt.legend()

        self._save_plot(msg)

    def plot_baro(self):
        msg = 'BARO'
        if self.verbose:
            print(f'Plotting '+msg+'...')

        df = self._get_msg(msg=msg)
        baro_groups = df.groupby('I')

        plt.figure(figsize=(7 * df['I'].nunique(), 2))

        for i, idx in baro_groups:
            idx = idx.reset_index(drop=True)
            self._save_csv(idx, msg+f'[{i}]')

            plt.subplot(1, df['I'].nunique(), i + 1)
            plt.plot(idx['TimeUS'] / 1e6, idx['Alt'], label=f'BARO[{i}].Alt]')
            plt.xlabel('Time [s]')
            plt.ylabel('Altitude [m]')
            plt.title(f'Delta: {idx['Alt'].max() - idx['Alt'].min():.1f} m')
            plt.grid(1)
            plt.legend()

        self._save_plot(msg)

    def plot_filter_review(self, 
                           target_instance:int=0, 
                           tune:bool=False, 
                           notch_freq=None, 
                           notch_bandwith=None,
                           notch_att=None,
                           autotune:bool=False):
        if self.verbose:
            print('Plotting Filter Review...')

        filter_reviewer = ArduPilotFilterReviewer(
            self.mavlog,
            tune=tune,
            notch_freq=notch_freq,
            notch_bandwith=notch_bandwith,
            notch_att=notch_att,
            autotune=autotune
            )

        filter_reviewer.plot_filter_review(target_instance=target_instance)
        if autotune:
            self._save_plot(f'FILTER_REVIEW_AUTOTUNE')
        else:
            self._save_plot(f'FILTER_REVIEW_TUNE_{tune}')  

    def save_summary(self):
        summary = self._generate_summary()

        filepath = f'summaries/{self.filedate}/SUMMARY_{self.filedate}.txt'

        with open(filepath, 'w') as f:
            f.write(f'ArduPilotLogReviewer Summary\n')
            f.write('='*40 + '\n\n')

            for line in summary:
                f.write(line + '\n')

        if self.verbose:
            print(f'Saving SUMMARY_{self.filedate}.txt...')

    def _get_msg(self, msg:str):
        df = pd.DataFrame(self.mavlog.get(msg).fields)
        if msg != 'PARM':
            df = df[(df['TimeUS'] / 1e6 >= self.T_MIN) & (df['TimeUS'] / 1e6 <= self.T_MAX)]
        else:
            df = df
        return df    
        
    def _detect_flight_window(self):
        msg = 'RCOU'
        df = pd.DataFrame(self.mavlog.get(msg).fields)

        pwm_min = self.parm.loc[self.parm['Name'] == f'SERVO{self.MOTOR_RCOU_CH[0]}_MIN', 'Value'].iloc[0]
        pwm_max = self.parm.loc[self.parm['Name'] == f'SERVO{self.MOTOR_RCOU_CH[0]}_MAX', 'Value'].iloc[0]
        pwm = df[f'C{self.MOTOR_RCOU_CH[0]}']

        time_us = df['TimeUS'].values / 1e6
        
        threshold = pwm_min + 0.1 * (pwm_max - pwm_min)
        above_threshold = pwm > threshold
        
        diff = np.diff(above_threshold.astype(int))
        rise_indices = np.where(diff == 1)[0] + 1  
        fall_indices = np.where(diff == -1)[0]    

        if len(rise_indices) == 0 or len(fall_indices) == 0:
            raise ValueError('Could not detect flight window')
        
        flight_start_idx = rise_indices[0]
        flight_end_idx = fall_indices[-1]
        
        T_MIN = time_us[flight_start_idx]
        T_MAX = time_us[flight_end_idx]
        
        if self.verbose:
            print(f'Flight Window Detected: {T_MIN:.1f}s - {T_MAX:.1f}s (Duration: {T_MAX - T_MIN:.1f}s)')
        
        return [T_MIN, T_MAX]         

    def _save_plot(self, name):
        if self.save_plots:
            if self.verbose:
                print(f'Saving Plot {name}_{self.filedate}.png...')
            filepath = f'plots/{self.filedate}/{name}_{self.filedate}.png'
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
        elif self.show_plots:
            plt.show()

    def _save_csv(self, df, name):
        if self.verbose:
            print(f'Saving CSV {name}_{self.filedate}.csv...')
        filepath = f'csvs/{self.filedate}/{name}_{self.filedate}.csv'
        df = df.copy()
        if 'TimeUS' in df.columns:
            df.rename(columns={'TimeUS': 'TimeS'}, inplace=True)
            df['TimeS'] /= 1e6

        df.to_csv(filepath, index=False)

    def _generate_summary(self):
        if self.verbose:
            print('Generating Summary...')

        summary_lines = []

        duration = self.T_MAX - self.T_MIN
        summary_lines.append(f'Flight Time: {duration:.1f} s')

        try:
            bat = self._get_msg('BAT')
            min_volt = bat['VoltR'].min()
            max_curr = bat['Curr'].max()
            mean_curr = np.mean(bat['Curr'])

            summary_lines.append(f'Minimum Battery Voltage: {min_volt:.2f} V')
            summary_lines.append(f'Maximum Battery Current: {max_curr:.2f} A')

            if self.ESC_BURST_A is not None and max_curr > self.ESC_BURST_A:
                summary_lines.append('WARNING: ESC Burst Current Exceeded!')
            elif self.ESC_CONT_A is not None and mean_curr > self.ESC_CONT_A:
                summary_lines.append('WARNING: ESC Continuous Current Exceeded!')

        except Exception as e:
            print(f'BAT Error: {e}')
            summary_lines.append('Battery Data Unavailable.')

        try:
            vibe = self._get_msg('VIBE')
            max_vibe = vibe[['VibeX', 'VibeY', 'VibeZ']].max().max()

            summary_lines.append(f'Max Vibration: {max_vibe:.2f} m/s/s')

            if max_vibe > 60:
                summary_lines.append('WARNING: Excessive Vibrations!')
            elif max_vibe > 30:
                summary_lines.append('WARNING: Vibrations In Caution Range!')

        except Exception as e:
            print(f'VIBE Error: {e}')
            summary_lines.append('Vibration Data Unavailable.')

        try:
            gps = self._get_msg('GPS')
            min_sats = gps['NSats'].min()
            max_hdop = gps['HDop'].max()

            summary_lines.append(f'Minimum Satellites: {min_sats}')
            summary_lines.append(f'Maximum HDOP: {max_hdop:.2f}')

            if min_sats < 12:
                summary_lines.append('WARNING: Low Satellite Count!')
            if max_hdop > 2:
                summary_lines.append('WARNING: Poor GPS Accuracy!')

        except Exception as e:
            print(f'GPS Error: {e}')
            summary_lines.append('GPS Data Unavailable.')

        try:
            msg = 'MAG'
            df = self._get_msg(msg=msg)
            mag_groups = df.groupby('I')

            rcou_ = self.rcou[['TimeUS', f'C{self.MOTOR_RCOU_CH[0]}']].copy()
            pwm_min = self.parm.loc[self.parm['Name'] == f'SERVO{self.MOTOR_RCOU_CH[0]}_MIN', 'Value'].iloc[0]
            pwm_max = self.parm.loc[self.parm['Name'] == f'SERVO{self.MOTOR_RCOU_CH[0]}_MAX', 'Value'].iloc[0]

            max_interferences = []

            for i, idx in mag_groups:
               idx = idx.reset_index(drop=True)

               mag_field = np.sqrt(idx['MagX']**2 + idx['MagY']**2 + idx['MagZ']**2)

               throttle_interp = np.interp(idx['TimeUS'], rcou_['TimeUS'], (rcou_[f'C{self.MOTOR_RCOU_CH[0]}'] - pwm_min) / (pwm_max - pwm_min) * 100)

               baseline_window = throttle_interp > 20
               if not np.any(baseline_window):
                   continue
               baseline_mag = mag_field[baseline_window].mean()

               mag_interference = ((mag_field - baseline_mag) / baseline_mag) * 100
               max_interference = mag_interference.max()
               max_interferences.append(max_interference)
            
            if max_interferences:
                summary_lines.append(f'Maximum Compass Interference: {max(max_interferences):.1f}%')
                if max(max_interferences) > 60:
                    summary_lines.append('WARNING: Compass Interference Is BAD (>60%)!')
                elif max(max_interferences) > 30:
                    summary_lines.append('WARNING: Compass Interference In Gray Zone (30–60%)!')
            else:
                summary_lines.append('WARNING: Compass Interference Data Unavailable!')

        except:
            summary_lines.append('Compass Interference Data Unavailable')

        try:
            df = self._get_msg('POWR')
            min_vcc = df['Vcc'].min()

            summary_lines.append(f'Min FCU Voltage: {min_vcc:.2f} V')

            if min_vcc < 5.0:
                summary_lines.append('WARNING: FCU Voltage Dropped Below Safe Level (5V)!')

        except Exception as e:
            print(f'POWR Error: {e}')
            summary_lines.append('FCU Voltage Data Unavailable')

        return summary_lines