
  

# A transfer learning approach for Mass Spectrometry Proteomics

  

This folder contains all of the code required to use CasanovoTL and reproduce the experimental results in our paper. The main model implementation is contained in the subdirectory Casanovo, which is forked from the publicly available [git repo](https://github.com/Noble-Lab/casanovo).

To avoid excessively long runtimes, we recommend running training and inference on a GPU if possible. 

## Requirements

Required packages are listed in the file requirements.txt.

We recommend using conda to manage dependencies. A new environment can be created with:

```
conda create --name casanovo_tl --file references.txt
```

Finally, you will need to navigate to the Casanovo directory, and install it with:

 ```
pip install .
```

## Evaluation

To run CasanovoTL to obtain spectrum representations for data in a given MGF file, run:

```
casanovo sequence <mgf_file> --model <pretrained_checkpoint> --config embed.yaml --output <output_file>
```

The embeddings for each spectrum will be saved to the file `<output_file>.pt`. The `embed.sh` scripts used to embed the data for each of our downstream tasks are  included in the subdirectory for each task. 

## Multi-task training

To perform multi-task training, you will need to download the de novo sequencing data originally used by casanovo from https://zenodo.org/records/12587317.

The subdirectory `multi_task` then contains the scripts necessary for running the multi-task training, and then using the trained model to embed data for each downstream task.  

## Downstream tasks
  
There is one subdirectory for each downstream task, containing the relevent code to embed spectra using CasanovoTL, train a task specific prediction head, and train the baseline models. The directory for the phospho task also contains the code for running the learning curve experiment.  

## Final Results

The jupyter notebook `plot_paper_figs.ipynb` contains code for plotting the final results shown in our paper, using saved results files output by each of the task-specific workflows. 

## Pre-trained Models and Data

The pre-trained CasanovoTL model checkpoint, along with the  checkpoint after finetuning, are available here: 
[https://drive.google.com/drive/folders/1ufpRD2sKvfBZY-am51ZTL_j14JqBeJLv?usp=drive_link](https://drive.google.com/drive/folders/1ufpRD2sKvfBZY-am51ZTL_j14JqBeJLv?usp=drive_link)
Additionally, this folder contains MGF files for the chimericity experiment, and the database search + GlyCounter results for the glyco data, as these are not already publicly available. The remaining datasets are already published on public repositories, as described in our manuscript. 
