for knn_weights in uniform distance; do 
  for Reorder_type in DKIndividual DKcriteria; do  
		for knn_neighbors in 2 3 4 5 6; do
                    python 'Main_classification_script.py' --knn_weights $knn_weights --knn_neighbors $knn_neighbors --Reorder_type $Reorder_type --FeatureComb_mode 'feat_comb7';
                done;
  done; 
done;
