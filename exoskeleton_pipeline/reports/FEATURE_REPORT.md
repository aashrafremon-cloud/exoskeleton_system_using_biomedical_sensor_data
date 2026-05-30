# FEATURE_REPORT

## Per channel (×11)
MAV, RMS, WL, VAR, ZC, SSC, IEMG (77 features total)

## DSP order
DC removal → HP 20 Hz → rectify → LP 6 Hz envelope → per-trial max normalization → decimate 100 Hz

## Future (not in v1 benchmark)
LOG, AAC, DASDV, MNF, MDF, spectral entropy, Hjorth, wavelets — add in v2 if R² plateaus
