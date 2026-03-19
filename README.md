<img width="3240" height="1030" alt="pipeline (1)" src="https://github.com/user-attachments/assets/feac6643-bdc0-4959-a9ac-18753b90f85e" />

## **Requirements**
```bash
conda env create -f environment.yml
```

## **Training Script**
- For running the training / inference script, see [template](https://github.com/AI4VSLab/OCTDiff/blob/main/Template-shell.sh)

## ** Dataset Curation **
The train/val/test set should by default in this format:
/Train
  /LowRes
  /HiRes
/Val
  /LowRes
  /HiRes
/Test
  /LowRes
  /HiRes

The [data_splitter](https://github.com/AI4VSLab/OCTDiff/blob/main/dataset_splitter.py)  is useful to curate such path structure.Please consider modifying the customized dataloader otherwise. 
To implement loss function with weights, a .csv file is needed.

## ** Citation **
```
@inproceedings{tian2025octdiff,
  title={OCTDiff: Bridged Diffusion Model for Portable OCT Super-Resolution and Enhancement},
  author={Tian, Ye and McCarthy, Angela and Gomide, Gabriel and Liddle, Nancy and Golebka, Jedrzej and Chen, Royce and Liebmann, Jeff and Thakoor, Kaveri A},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
