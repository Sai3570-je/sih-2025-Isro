
====================================================
   GNSS ERROR FORECASTING - KALMAN FILTER PIPELINE
====================================================

Satellites Processed: 2
Training Period: Days 1-7
Forecast Period: Day 8 (96 x 15-min intervals)

OUTPUT FILES:
  - outputs/predictions_day8_geo.csv
  - outputs/predictions_day8_meo.csv
  - models/kalman_params_*.pkl
  - results/metrics_summary.json
  - results/figures/*.png

HOW TO RUN:
  python main.py --mode all --data-folder data

Generated: 2025-11-27T09:44:11.680513
