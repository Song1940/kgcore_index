# kgcore_index
This is the implementation of kg-core index, which is described in the following papaer:
- Efficient Co-occurrence-based Core Decomposition in hypergraphs: an Index-based Approach

## How to use?
### A simple description for indexing tree construction and query processing for each index are as follows:

### Index construction
- There are four types of indexing tree
  - Naive(naive)
  - Horizontal(hori)
  - Vertical(vert)
  - Diagonal(diag)
[command for the tree type is in ()]
- Input parameters
  - Path of the hypergraph data
  - Index type
-Then '.pkl' of indexing tree file automatically saved in your working directory  
Example code for the index construction is below
```
python index_construction.py --file_path scr/network.hyp --type naive
# naive.pkl file will be stored in your working directory within few seconds(minutes)
```

### Query processing
- With the indexing tree you just builts, you can query to achieve (k,g)-core of the hypergraph
- Input parameters
  - Path for the indexing tree(path of .pkl file)
  - Index type
  - k
  - g
- This will return
  - the size of core(number of nodes in the core you queried)
  - every each node

Example code for the index construction is below
```
python query_processing.py --file_path scr/naive.pkl --type naive --k 5 --g 5

```
