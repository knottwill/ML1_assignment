# M1 Coursework - William Knottenbelt

This repository contains all code necessary to reproduce all results discussed in `report/main.pdf`.

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

The data required for the project is saved in `data/`.

The solution to each question is in a script titled `solve_[question-number]_[parts].py`. For example, running `solve_3_abc.py` will yield the solutions to question 3 part a, b and c. Plots will be saved in the `plots/` directory and any additional processed data will be saved in `data/`. Other results will be printed to the terminal.

Below are the commands to generate all the results in the project. THE SCRIPTS MUST BE RUN IN THE ORDER GIVEN (certain scripts depend on pre-processed data from earlier scripts).

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
