####
# Obtain spectrum embeddings for all of the downsampled phospho datasets
####
for size in "1024" "512" "256" "128" "64" "32" "16" "8" "4" "2" "1"; do 
    for dataset in data/$size/*/; do
        mkdir embeddings/$size
        mkdir embeddings/$size/$(basename "$dataset")
        casanovo sequence "$dataset"phospho.mgf --model casanovo_massivekb_v4_0_0.ckpt --config embed.yaml --output embeddings/$size/$(basename "$dataset")/phospho.mztab
        casanovo sequence "$dataset"non_phospho.mgf --model casanovo_massivekb_v4_0_0.ckpt --config embed.yaml --output embeddings/$size/$(basename "$dataset")/non-phospho.mztab
    done
done