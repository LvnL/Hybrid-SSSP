# Hybrid-SSSP
This project combines the advantages that CPUs and GPUs have for graph processing. The Bellman-Ford algorithm is used with selection of CPU or GPU being done on a per iteration basis. Both the CSR and COO formats are used to implement efficient processing for the CPU and GPU, respectively. The project was inspired by [Garaph](https://www.usenix.org/system/files/conference/atc17/atc17-ma.pdf) and was developed as part of [CS:4700 - High Performance & Parallel Computing](https://myui.uiowa.edu/my-ui/courses/details.page?ci=158677&id=990815)
## Prerequisites
- `make` to use the `Makefile`
- `g++` and `nvcc` for compilation
- https://snap.stanford.edu/data/higgs-social_network.edgelist.gz
## Usage
- Run `make` or `make runner` from the root directory
- Execute the resulting file while passing the `.edgelist` file (for example `$ ./runner higgs-social_network.edgelist`)
