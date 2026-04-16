import argparse, os
from ardupilot_log_reviewer import ArduPilotLogReviewer

def main():
    parser = argparse.ArgumentParser(description="ArduPilot Log Reviewer CLI")
    
    parser.add_argument("filedate", help="Date of ArduPilot DataFlashLog (YYYY-MM-YY)")
    parser.add_argument("--tmin", type=float, help="Start Time (seconds)")
    parser.add_argument("--tmax", type=float, help="End Time (seconds)")
    parser.add_argument("--motor-channels", type=int, nargs='+', default=[10, 11, 12], help='Motor Channels')
    parser.add_argument("--esc-cont-limit", type=int, default=70, help="ESC Continuous Current Limit")
    parser.add_argument("--esc-burst-limit", type=int, default=80, help="ESC Burst Current Limit")

    args = parser.parse_args()

    log_path = os.path.join("logs", f"{args.filedate}.BIN")

    reviewer = ArduPilotLogReviewer(
        filedate=args.filedate,
        filepath=log_path,
        MOTOR_RCOU_CH=args.motor_channels,
        auto_detect_flight=False,
        T_MIN=args.tmin,
        T_MAX=args.tmax,
        ESC_CONT_A=args.esc_cont_limit,
        ESC_BURST_A=args.esc_burst_limit,  
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

if __name__ == "__main__":
    main()