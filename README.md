# ArduPilot Log Reviewer

A Python-based tool for automated analysis and visualisation of ArduPilot DataFlash logs.

## Features

- **Automated Flight Detection:** Detects flight start and end times based on motor output.
- **Plotting:**
  - **ATT:** Roll, Pitch, and Yaw with RMSE (Root Mean Squared Error) calculations.
  - **VIBE:** Vibrations for all IMU instances with vibration limits.
  - **BAT:** Voltage and current monitoring with ESC current and voltage limits.
  - **MAG:** Compass interference analysis.
  - **GPS & BARO:** Signal quality with quality limits and altitude tracking.
  - **Filter Review:** Harmonic notch filter simulation and PSD (Power Spectral Density) analysis.
- **Export:** Automatically saves processed data to .csv and plots to .png.
- **Summary:** Produces a .txt flight summary with warnings for exceeded limits.

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `pymavlog`

## Usage

````python
from ArduPilotLogReviewer import ArduPilotLogReviewer

filedate = "YYYY-MM-DD"
filepath = f"logs/{filedate}.BIN"
MOTOR_RCOU_CH = [1, 2, 3]

reviewer = ArduPilotLogReviewer(
    filedate=filedate,
    filepath=filepath,
    MOTOR_RCOU_CH=MOTOR_RCOU_CH,
    auto_detect_flight=True,
    save_plots=True,
    show_plots=False
)

reviewer.plot_att()
reviewer.plot_vibes()
reviewer.plot_rcou()
reviewer.plot_esc()
reviewer.plot_bat()
reviewer.plot_imus()
reviewer.plot_powr()
reviewer.plot_compass_interference()
reviewer.plot_gps()
reviewer.plot_baro()
reviewer.save_summary()
reviewer.plot_filter_review()```
