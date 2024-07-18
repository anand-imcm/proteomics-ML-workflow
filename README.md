# proteomics-ML-workflow

> [!WARNING]
> This project is under development and is not ready for production use.

## todo

- [x] Initial Layout
- [ ] Workflow description
- [ ] Describe Inputs
- [ ] Describe Outputs
- [ ] Usage instructions
- [ ] List all the components and their licenses


## Workflow Steps

- **Preprocessing** : Performs Z-score normalization on the input data.
- **Classification** : TODO

## Inputs

The main inputs to the workflow are:

- **required**
  - `input_csv` : Input file in .csv format.
  - `output_prefix` : Sample name. This will be used as prefix for all the output files.
- **optional**
  - `preprocessing.cpu` : [Integer] Total number of CPUs to be used in the `preprocessing` step. Default value: `8`
  - `preprocessing.memory_gb`: [Integer] Total number of RAM to be used in the `preprocessing` step. Default value: `8`

## Outputs

The main output files are listed below:

- **Preprocessing**
  - `processed_csv` : The proecessed ouput file in .csv format.
  
## Components

| Package | License |
|---------|---------|
| [micromamba==1.5.5](www.github.com/mamba-org/mamba#micromamba) | BSD-3-Clause |
| [python==3.10](www.python.org/) | PSF/GPL-compat |
| [joblib==1.4.2](www.github.com/joblib/joblib) | BSD-3-Clause |
| [matplotlib==3.9.1](www.matplotlib.org) | PSF/BSD-compat |
| [numpy==2.0.0](www.numpy.org/) | BSD |
| [pandas==2.2.2](www.pandas.pydata.org/) | BSD 3-Clause License |
| [scikit-learn==1.5.1](www.scikit-learn.org) | BSD-3-Clause |
