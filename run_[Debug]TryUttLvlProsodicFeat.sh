#for feature_method in Disvoice_phonation Disvoice_prosody_energy prosodyF0;do
for feature_method in Disvoice_prosodyF0;do
  for filepath in data/Segmented_ADOS_TD_normalized data/Segmented_ADOS_normalized;do
    python \[Debug\]TryUttLvlProsodicFeat.py --filepath $filepath --method $feature_method; 
  done; 
done;
