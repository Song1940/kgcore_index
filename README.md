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
 
Example code for the index construction is below
```
python index_construction.py --file_path scr/network.hyp --type naive
# naive.pkl file will be stored in your working directory within few seconds(minutes)
```

### Query
