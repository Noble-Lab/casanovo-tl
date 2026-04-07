###
# Run inference with the multi-task trained model for data from each task
###

# Multi-task model checkpoint 
checkpoint='joint_trained_casanovo_foundation.ckpt'

# Phospho 
casanovo sequence ../phospho/data/non_phospho_test_combined_32.mgf --model $checkpoint --output ../phospho/embeddings_multi/non_phospho_test.mztab
casanovo sequence ../phospho/data/phospho_test_combined_32.mgf --model $checkpoint --output ../phospho/embeddings_multi/phospho_test.mztab

casanovo sequence ../phospho/data/non_phospho_train_combined_32.mgf --model $checkpoint --output ../phospho/embeddings_multi/non_phospho_train.mztab
casanovo sequence ../phospho/data/phospho_train_combined_32.mgf --model $checkpoint --output ../phospho/embeddings_multi/phospho_train.mztab

casanovo sequence ../phospho/data/non_phospho_val_combined_32.mgf --model $checkpoint --output ../phospho/embeddings_multi/non_phospho_val.mztab
casanovo sequence ../phospho/data/phospho_val_combined_32.mgf --model $checkpoint --output ../phospho/embeddings_multi/phospho_val.mztab

# Quality 
casanovo sequence ../quality/data/train_pos.mgf --model $checkpoint --output ../quality/embeddings_multi/train_pos.mztab
casanovo sequence ../quality/data/train_neg.mgf --model $checkpoint --output ../quality/embeddings_multi/train_neg.mztab

casanovo sequence ../quality/data/test_pos.mgf --model $checkpoint --output ../quality/embeddings_multi/test_pos.mztab
casanovo sequence ../quality/data/test_neg.mgf --model $checkpoint --output ../quality/embeddings_multi/test_neg.mztab

casanovo sequence ../quality/data/valid_pos.mgf --model $checkpoint --output ../quality/embeddings_multi/val_pos.mztab
casanovo sequence ../quality/data/valid_neg.mgf --model $checkpoint --output ../quality/embeddings_multi/val_neg.mztab


# Chimera 
casanovo sequence ../chimera/data/train_single.mgf --model $checkpoint --output ../chimera/embeddings_multi/single_train.mztab
casanovo sequence ../chimera/data/train_chimeric.mgf --model $checkpoint --output ../chimera/embeddings_multi/chimeric_train.mztab

casanovo sequence ../chimera/data/test_single.mgf --model $checkpoint --output ../chimera/embeddings_multi/single_test.mztab
casanovo sequence ../chimera/data/test_chimeric.mgf --model $checkpoint --output ../chimera/embeddings_multi/chimeric_test.mztab

casanovo sequence ../chimera/data/val_single.mgf --model $checkpoint --output ../chimera/embeddings_multi/single_val.mztab
casanovo sequence ../chimera/data/val_chimeric.mgf --model $checkpoint --output ../chimera/embeddings_multi/chimeric_val.mztab

# Glyco 
casanovo sequence ../glyco/data/train_pos.mgf --model $checkpoint --output ../glyco/embeddings_multi/train_pos.mztab
casanovo sequence ../glyco/data/train_neg.mgf --model $checkpoint --output ../glyco/embeddings_multi/train_neg.mztab

casanovo sequence ../glyco/data/test_pos.mgf --model $checkpoint --output ../glyco/embeddings_multi/test_pos.mztab
casanovo sequence ../glyco/data/test_neg.mgf --model $checkpoint --output ../glyco/embeddings_multi/test_neg.mztab

casanovo sequence ../glyco/data/val_pos.mgf --model $checkpoint --output ../glyco/embeddings_multi/val_pos.mztab
casanovo sequence ../glyco/data/val_neg.mgf --model $checkpoint --output ../glyco/embeddings_multi/val_neg.mztab
