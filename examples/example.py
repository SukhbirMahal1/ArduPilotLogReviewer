from ardupilot_log_reviewer.ArduPilotLogReviewer import ArduPilotLogReviewer

filedate = "2026-03-26"
filepath = f"logs/{filedate}.BIN"
MOTOR_RCOU_CH = [10, 11, 12]

reviewer = ArduPilotLogReviewer(
    filedate=filedate,
    filepath=filepath,
    MOTOR_RCOU_CH=MOTOR_RCOU_CH,
    auto_detect_flight=False,
    T_MIN=2675,
    T_MAX=2800,
    ESC_CONT_A=70,
    ESC_BURST_A=80,
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
reviewer.plot_filter_review()
reviewer.plot_filter_review(tune=True, notch_freq=80, bw=80/2, att=10)
filedate = "2026-03-26"
filepath = f"logs/{filedate}.BIN"
MOTOR_RCOU_CH = [10, 11, 12]

reviewer = ArduPilotLogReviewer(
    filedate=filedate,
    filepath=filepath,
    MOTOR_RCOU_CH=MOTOR_RCOU_CH,
    auto_detect_flight=False,
    T_MIN=2675,
    T_MAX=2800,
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
reviewer.plot_filter_review()
reviewer.plot_filter_review(tune=True, notch_freq=80, bw=80/2, att=10)