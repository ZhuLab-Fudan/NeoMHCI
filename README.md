# NeoMHCI

NeoMHCI is a deep learning tool designed for the prediction of multi-allele and single-allele MHC-I ligand presentation as well as neoepitope immunogenicity.

All data used in this research is available for free download [here]().

## Requirements

The following packages and their versions are required:

* python == 3.8.8
* pytorch == 1.7.1
* numpy == 1.19.2
* click == 7.1.2
* ruamel.yaml == 0.16.12
* tqdm == 4.56.0
* fair-esm == 0.4.2

## Usage

The python script `main.py` can execute two types of task modes: `test` and `motif`.

### Test

Example command:

```shell
python main.py -d ./config/data.yaml -m ./config/model.yaml --mode test -s 0 -n 10 -g 0
```

In `data.yaml`, the following data definitions exist:

* `mhc_seq`: Defines the pseudo sequence for each MHC-I molecule.
* `cell_line`: Defines the real MHC-I molecules corresponding to multi-allele or single-alele names.
* `test`: Data to be tested.
* `rawdata`: Defines the hundred thousand random peptides used for drawing motifs.

When testing new single allele data or multi-allele data, ensure the data format is consistent with that in `./data/example_data.txt`. Additionally, ensure the new MHC-I molecule or combinations are listed in the file corresponding to `cell_line` (default `./data/allelelist`), and that the 34-mer pseudo sequence for the new MHC-I molecule is provided in the file corresponding to `mhc_seq` (default `./data/MHC_pseudo.dat`).

The output results will be stored in `./outputs`.

### Motif

Example command:

```shell
python main.py -d ./config/data.yaml -m ./config/model.yaml --mode motif -s 0 -n 10 -g 0 --allele CPH-08-TISSUE --motif_len 9
```

You can specify the single allele or multi-allele to output the top-1% (1000 peptides) by setting the `--allele` option. The `--motif_len` option defines the length of the chosen motif, with a maximum length of 15.

The output results will be placed in `./motifs`.

## Declaration

This tool is free for non-commercial use. For commercial use, please contact Mr. Wei Qu and Prof. Shanfeng Zhu (zhusf@fudan.edu.cn).

