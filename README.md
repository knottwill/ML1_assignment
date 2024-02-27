# Machine Learning 1 - Coursework Assignment

This project analyses the sparse, high-dimensional datasets found in `data/` using cluster analysis, classification, regression, outlier detection, imputation and dimensionality reduction. We utilised the following techniques: k-Nearest Neighbors, Random Forest, Logistic Regression with Lasso regularisation, K-Means, Spectral Clustering, PCA and UMAP. 

This repository (codebase and report in `report/main.pdf`) was submitted in fulfillment of the coursework assignment detailed in `assignment.pdf`. This was one of three assignments completed in the first semester of my Master's in Data Intensive Science at the University of Cambridge. 

My submission recieved a grade of 84% (may be subject to moderation in Final Examiners' Meeting in September 2024).

### Installation

Clone the repository:

```bash
$ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/wdk24.git
$ cd wdk24
```

To generate the docker image and run the container, navigate to the root directory and use:

```bash
$ docker build -t <image_name> .
$ docker run -ti <image_name>
```

Alternatively, clone the conda environment in `environment.yml`.

### Usage

The data required for the project is saved in `data/` (including processed data which is created by running the scripts).

The solutions to the questions are found by running scripts titled `solve_[question-number]_[parts].py`. For example, running `solve_3_abc.py` will yield the solutions to question 3 part a, b and c. Plots will be saved in the `plots/` directory and any additional processed data will be saved in `data/`. Other results will be printed to the terminal.

Below are the commands to generate all the results in the project. The scripts should be run in the order given below, since certain scripts depend on pre-processed data from earlier scripts (although these datasets were included in the repository just in case).

```bash
$ python src/solve_1_a.py    
$ python src/solve_1_b.py    
$ python src/solve_1_cde.py     
$ python src/solve_2.py      
$ python src/solve_3_abc.py  # 20s   
$ python src/solve_3_de.py    
$ python src/solve_4_b.py    # 1m 50s   
$ python src/solve_4_c.py   
$ python src/solve_4_de.py  
$ python src/solve_4_f.py   
$ python src/solve_5_a.py    # 2m  
$ python src/solve_5_bc.py   # 15s   
```

The module `src/utils.py` contains extra functionality which is re-used in various scripts.  

### Machine Specifications and Timing

I ran all scripts on my personal laptop with the following specifications:
- Chip:	Apple M1 Pro
- Total Number of Cores: 8 (6 performance and 2 efficiency)
- Memory (RAM): 16 GB
- Operating System: macOS Sonoma v14.0

Timing for scripts
- `solve_3_abc.py` took 20s
- `solve_4_b.py` took 1 minute 50 seconds
- `solve_5_a.py` took 2 minutes
- `solve_5_bc.py` took 15 seconds
- All other scripts ran in less than 10 seconds
