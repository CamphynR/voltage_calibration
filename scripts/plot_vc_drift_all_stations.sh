for STATION in 11 12 13 21 22 23 24; do
    echo $STATION;
    python scripts/plot_vc_drift_single.py "pickles/vc_drift_ref/vc_drift_ref_s${STATION}_w-0.2_0.2"
done
