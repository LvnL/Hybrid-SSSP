# Hybrid-SSSP
This project combines the advantages that CPUs and GPUs have for graph processing. The Bellman-Ford algorithm is used with selection of CPU or GPU being done on a per iteration basis. Both the [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) and [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) formats are used to implement efficient processing for the CPU and GPU, respectively. The project was inspired by [Garaph](https://www.usenix.org/system/files/conference/atc17/atc17-ma.pdf) and was developed as part of [CS:4700 - High Performance & Parallel Computing](https://myui.uiowa.edu/my-ui/courses/details.page?ci=158677&id=990815)
<br>
<br>
[Google Doc](https://docs.google.com/document/d/1TDb8Q4xWwHBhfGl76xbntCk3CzSr8P6IUSqzjJBTcAM/edit)
## Prerequisites
- `make` to use the `Makefile`
- `g++` and `nvcc` for compilation
- https://snap.stanford.edu/data/higgs-social_network.edgelist.gz
## Usage
- Run `make` or `make runner` from the root directory
- Execute the resulting file while passing the `.edgelist` file (for example `$ ./runner higgs-social_network.edgelist`)
