for knn_weights in uniform distance; do 
  for Reorder_type in DKIndividual; do  
		for knn_neighbors in 4 5 6; do
                    python 'Main_classification_script.py' --knn_weights $knn_weights --knn_neighbors $knn_neighbors --Reorder_type $Reorder_type --FeatureComb_mode 'Comb_staticLOCDEP_dynamicLOCDEP_dynamicphonation';
                done;
  done; 
done;
