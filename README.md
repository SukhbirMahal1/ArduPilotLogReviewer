# ArduPilot Log Reviewer

![python](https://img.shields.io/badge/python-3.9--3.12-blue)
![jupyter](https://img.shields.io/badge/jupyter-notebook-orange)

A Python-based tool for automated analysis and visualisation of ArduPilot DataFlash logs.

## Installation

![Terminal](https://badgen.net/badge/icon/terminal?icon=terminal&label)
[![git](https://badgen.net/badge/icon/git?icon=git&label)](https://git-scm.com)

Clone the GitHub repository:

```bash
git clone https://github.com/SukhbirMahal1/ArduPilotLogReviewer.git
cd ArduPilotLogReviewer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as an editable package:
```bash
pip install -e .
```

## Usage

Normal log review:

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
```

Filter review:

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
)

reviewer.plot_filter_review()
```
