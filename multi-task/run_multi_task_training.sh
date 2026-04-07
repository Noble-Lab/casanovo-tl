###
# Multi-task fine tuning of the pre-trained casanovo model
###

casanovo train data/casanovo_train.mgf -p data/casanovo_valid.mgf --model casanovo_massivekb_v4_0_0.ckpt --config config.yaml
