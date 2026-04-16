# ArduPilot Log Reviewer

A Python-based tool for automated analysis and visualisation of ArduPilot DataFlash logs.

## Installation

Installation with pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from ardupilot_log_reviewer import ArduPilotLogReviewer

filedate = "YYYY-MM-DD"
filepath = "path/to/your/log.BIN"
MOTOR_RCOU_CH = []

reviewer = ArduPilotLogReviewer(
    filedate=filedate,
    filepath=filepath,
    MOTOR_RCOU_CH=MOTOR_RCOU_CH,
    auto_detect_flight=True,
    save_plots=True,
    show_plots=True
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
reviewer.plot_filter_review()
```
