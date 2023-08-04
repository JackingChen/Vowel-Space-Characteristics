Alignment_root='Alignment_ADOShappyDAAIKidallDeceiptformosaCSRC_chain'


for Alignment_root in Alignment_DAAIKidFullDeceptCSRCformosa_all_Trans_ADOS_train_happynvalid_langMapped_chain ;do
	trnpath=/mnt/sdd/jackchen/egs/formosa/s6/${Alignment_root}/new_system/kid_cmpWithHuman/ADOS_tdnn_fold_transfer
	for wind in 1 2 3 4 5;do 		
		python Compare_human_Auto.py --PoolFormantWindow $wind --trnpath $trnpath; 
	done; 

	python Compare_human_Auto.py  --avgmethod 'mean'  --trnpath $trnpath
done;
