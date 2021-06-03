A Joint Convolutional and Space Quad-Directional LSTM Network for Phase Unwrapping
==============================

<p align="center">
  <img src="https://github.com/Laknath1996/DeepPhaseUnwrap/blob/main/reports/figures/summary.jpg">
</p>

This repository contains the source code for the deep neural arcihetcure proposed by the ICASSP 2021 paper titled ["A Joint Convolutional and Space Quad-Directional LSTM Network for Phase Unwrapping"](https://ieeexplore.ieee.org/document/9414748). 

If you use this code/paper for your research, please consider citing,

```
@INPROCEEDINGS{9414748,  
author={Perera, Malsha V. and De Silva, Ashwin},  
booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},   
title={A Joint Convolutional and Spatial Quad-Directional LSTM Network for Phase Unwrapping},   
year={2021},  
volume={},  
number={},  
pages={4055-4059},  
doi={10.1109/ICASSP39728.2021.9414748}}
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Datasets created/ used by the project   
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- A tutorial on the project 
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   ├── models         <- Scripts to define models and losses.
    |   └── visualization  <- Scripts to create plots
    ├── create_synthetic_phase_dataset.py <- Create datasets
    ├── train_model.py                    <- Train models
    └── test_model.py                     <- Test models
--------

## Installation Guide

### Step 1 : Clone the Repository

Clone the repository using the following command.

````
$ git clone https://github.com/Laknath1996/DeepPhaseUnwrap.git
````

### Step 2 : Install Dependencies

Use the `requirements.txt` file given in the repository to install the dependencies via `pip`.

````
$ pip install -r requirements.txt 
````

### Step 3 : Install Dependencies

Use the `create_synthetic_phase_dataset.py`, `train_model.py` and `test_model.py` files to create phase datasets, train models, and validate them, respectively.          

## Tutorial

`notebooks/tutorial.ipynb` describes the specifics and the execution steps of the network. 

## Authors

* **Ashwin De Silva** - ashwindesilva@yahoo.com
* **Malsha Perera** - malshaperera14@gmail.com

At the time of this work, both the authors were with the Department of Electronics and Telecommunication Engineering, University of Moratuwa, Sri Lanka.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Biomedical Engineering Laboratory, Dept. of Electronic and Telecommunication Eng., University of Moratuwa, Sri Lanka.

## References

[[1]] M. V. Perera and A. De Silva, "A Joint Convolutional and Spatial Quad-Directional LSTM Network for Phase Unwrapping," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 4055-4059, doi: 10.1109/ICASSP39728.2021.9414748.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
