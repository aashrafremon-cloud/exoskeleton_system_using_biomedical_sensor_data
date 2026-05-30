# CHANNEL_AUDIT

## Training & API (correct)
All **11** Camargo EMG channels:
gastrocmed, tibialisanterior, soleus, vastusmedialis, vastuslateralis,
rectusfemoris, bicepsfemoris, semitendinosus, gracilis, gluteusmedius, rightexternaloblique

## example_config_data.csv (NOT used for training)
Only 4 example amplitudes: bicepsfemoris, rectusfemoris, semitendinosus, vastusmedialis.
This file is documentation/demo config — **not** the training feature set.

## Resolution
- Keep 11-channel API input `(N, 11)` raw EMG @ ≥200 samples
- Do NOT reduce to 4 channels without retraining
