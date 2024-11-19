### About

### Aims

### Directory Structure
"OriginalData/" contains the inputs to the process, including lists of genes of interest and the gene-expression measurements.
"ProcessedData/" contains the output results of the various scripts.
"SerializedObjects/" contains various intermediate python objects. Due to the size of the resulting objects, the contents of this folder are not uploaded to github.
"Images/" contains the various image outputs.

### Process
1. Run "DataClustering.py" to compute the different gene clusters and to create the formatted datasets.
2. Run "AggregateClusterData.py" to aggregate all measurements within a cluster together.
3. Run "CalibrateGP.py" to:
	1. Compute Gaussian process interpolations of each cluster
	2. Sample mean of process
	3. Compute spline interpolation of mean
	4. Creates diagnostic images for the gaussian processes.