# SYNC_REPORT

## Rates
- EMG: 1000 Hz (raw) → 100 Hz after decimation (factor 10)
- IK: 200 Hz (Camargo) → 100 Hz after decimation (factor 2)
- Duration sync: `min(len_emg/1000, len_ik/200)` then trim both

## Window mapping
1. EMG windows: 20 samples @ 100 Hz = 200 ms, overlap 50% → step 10 samples (100 ms)
2. IK downsampled to 100 Hz, same window grid
3. **Target** for window starting at sample `s`: `knee_angle[s + 19 + lag_samples]` on 100 Hz grid
4. `lag_samples = round(lag_ms / 1000 * 100)` — EMG features lead knee response

## Alignment guarantee
Both EMG features and IK targets use identical window start indices after resampling to 100 Hz.
