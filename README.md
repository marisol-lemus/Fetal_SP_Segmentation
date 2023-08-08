# Fetal_SP_Segmentation
![](figure/sp_example.png)

## Table of contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Container](#container)
* [Installation](#installation)
* [Training](#training)
* [Evaluation](#evaluation)
* [Performance](#performance)



## Overview
Deep learning U-Net to predict the subplate (SP), cortical plate (CP) and inner plate segmentation from fetal MRI scans. 


## Dataset

The model was trained with a dataset of 89 MRI's of subjects between 22 GW and 31 GW. To train the model, the data splits for training/validation/testing are given in ./data/. A gestational age (GA) file is needed that includes the GA of the subjects in alphabetical order. You can find an example in ./data/GA.txt

If you are at the FNDDSC laboratory, you can access the data information here
``` bash
/neuro/labs/grantlab/research/MRI_processing/marisol.lemus/subplate_seg_deep_project/dataset/SP_data_information.csv
```

## Container

You can run the pipeline in a singularity file. This will work on any computer and is simple to set up.  

You can build a singularity image with the following command. You can read more [here](https://docs.sylabs.io/guides/3.5/user-guide/definition_files.html). Make sure to download the [weights](https://bit.ly/sp-segmentation-weights) first!

``` bash
sudo singularity fetal_sp_seg.sif container.def;
```

Once you have your container, you need to make a directory to hold the images you want to analyze and the results and create a variable to identify the subject. For example:

``` bash
mkdir subj1;
cp T2.nii.gz ./subj1;
file=subj1;
```
The T2.nii.gz file is the reconstruction of your subjects. You can use any names for the images and the directory, but you'll have to modify sightly the next command. 

``` bash
singularity run --no-home -B ${file}/:/data ./fetal_sp_seg_noatt.sif T2.nii.gz ./output 0;
``` 

If you are part of the FNNDSC laboratory, the docker usage is like this:
``` bash
singularity run --no-home -B ${file}/:/data /neuro/labs/grantlab/research/MRI_processing/marisol.lemus/subplate_seg_deep_project/code/noatt_code/fetal_sp_seg_noatt.sif recon_to31_nuc.nii ./ 0
``` 
The results will be saved on the same $file path, and you'll use the gpu 0 of the machine. The variable file should be the identification of an MRI subject where the reconstruction (recon_to31_nuc.nii) is located.  Here is an example:

``` bash
file=1234567/2008.04.06-031Y-MR_Fetal_Body_Indications-12345/;
```

## Installation

This implementation mainly relies on [Python](https://www.python.org/) . The dependencies can installed by running the following code: 
``` bash
python -m venv SP_env
source SP_env/bin/activate
pip install -r requirements.txt
``` 

## Training 

The following instruction shows how to train the subplate model on a given dataset. 

First, you need to generate the tensorflow (TF) Records, which is a file format used to store the data in binary to be used later on the training code. 
``` bash
nohup python3  code/fetal_subplate_seg_records.py -infol_MR ./data/MR -infol_GT ./data/GT -wl ./tf_records/  -fe 5 -all -sm skf -fi ./data/GA  -gpu 0 -f 5 -bs 30 -fp >tf_records_noatt.out &
```
 Where -infol_MR is the path with the ground truth data, -wl is the path where the TF records will be saved.  Please refer to config.py for detailed configurations.  

 Then, you need to run the following code that trains a U-net network for 100 epochs. 

 ``` bash
nohup python3  code/fetal_subplate_seg.py -infol_MR ./data/MR -infol_GT ./data/GT -infol_rec ./tf_records -wl ./weights/ -hl ./history/ -fe 5 -all -sm skf -fi ./data/GA  -gpu 0 -f 5 -bs 30 -opt SGD -lr 0.0001 -l asymmetric_focal_tversky_loss >weights_noatt.out&
 ```
To visualize the results run the following commands. The first code will show the dice coefficient and losses graphs of the Kfolds, while the second code will show the average dice coefficient and loss of both training and validation. 

``` bash
cd history/
python3 ./results/plots.py . ;
python3 ./results/avg_score2.py . ;
```

## Evaluation

The model weights can be downloaded from this [link](https://bit.ly/sp-segmentation-weights). To predict the subplate for an MRI reconstruction using the weights, please run

``` bash
python code/SP_segmentation.py -input evaluation/subj1/recon_to31_nuc.nii -output evaluation/subj1/ -axi ../pretrained/axi.h5 -cor ../pretrained/cor.h5 -sag ../pretrained/sag.h5;
```
Where -input is MRI reconstruction to segment, -output is the path where the segmentation will be saved . Please refer to config.py for detailed configurations.  


## Performance
The model was evaluated using cross-validation. The final training had a training dice coefficient of 0.94 and a training loss of 0.52. The validation dice coefficient was 0.94 with a loss of 0.52

![](figure/predictions.png)


