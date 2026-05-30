# Model Explainability (RandomForest feature importance)

SHAP optional (`pip install shap` in a clean venv). This report uses Gini importance.

## Muscles ranked by total importance

- **gastrocmed**: 0.6011
- **tibialisanterior**: 0.1257
- **gluteusmedius**: 0.0967
- **soleus**: 0.0511
- **gracilis**: 0.0293
- **semitendinosus**: 0.0216
- **vastusmedialis**: 0.0215
- **rightexternaloblique**: 0.0177
- **vastuslateralis**: 0.0171
- **rectusfemoris**: 0.0123
- **bicepsfemoris**: 0.0056

## Top 20 features

- `gastrocmed_iemg`: 0.3592
- `gastrocmed_mav`: 0.2250
- `gluteusmedius_iemg`: 0.0416
- `gluteusmedius_mav`: 0.0415
- `tibialisanterior_wl`: 0.0357
- `tibialisanterior_iemg`: 0.0348
- `tibialisanterior_mav`: 0.0295
- `soleus_iemg`: 0.0153
- `soleus_mav`: 0.0138
- `gracilis_var`: 0.0111
- `soleus_var`: 0.0107
- `tibialisanterior_rms`: 0.0100
- `rightexternaloblique_rms`: 0.0096
- `tibialisanterior_ssc`: 0.0090
- `gastrocmed_wl`: 0.0075
- `gastrocmed_var`: 0.0070
- `soleus_rms`: 0.0070
- `tibialisanterior_var`: 0.0067
- `gracilis_wl`: 0.0064
- `vastusmedialis_rms`: 0.0062