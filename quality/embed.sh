####
# Obtain spectrum representations for all spectra in the quality task
####
mkdir embeddings
for dataset in data/*.mgf; do
    casanovo sequence $dataset --model casanovo_massivekb_v4_0_0.ckpt --config embed.yaml --output embeddings/$(basename "$dataset").mztab
done
