import networkx as nx
import time
import os
import argparse
import copy
from types import MappingProxyType
import pickle
from pympler import asizeof
import operator

class TreeNode:
    def __init__(self, name):
        self.aux ={}
        self.name = name
        self.children = []
        self.next = None
        self.value = set()
        self.jump = None

def load_hypergraph(file_path):
    hypergraph = nx.Graph()  # Create an empty hypergraph
    E = list()

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # Use set to ignore duplicate values in each line and strip whitespace from node names
            nodes = {node.strip() for node in line.strip().split(',')}
            nodes = {int(x) for x in  nodes}
            hyperedge = set(nodes)  # Use frozenset to represent the hyperedge
            E.append(hyperedge)
            for node in nodes:
                if node not in hypergraph.nodes():
                    hypergraph.add_node(node, hyperedges=list())  # Add a node for each node
                hypergraph.nodes[node]['hyperedges'].append(hyperedge)  # Add the hyperedge to the node's hyperedge set

    return hypergraph, E

def querying_for_naive_index(tree, k,g):
    max_g = len(tree.children)
    if g > max_g:
        return None
    max_k = len(tree.children[g - 1].children)
    if k > max_k:
        return None
    else:
        core = tree.children[g-1].children[k-1].value

    return core

def querying_for_one_level(tree, k,g):
    max_g = len(tree.children)
    if g > max_g:
        return None
    max_k = len(tree.children[g - 1].children)
    if k > max_k:
        return None
    else:
        core = set()
        header = tree.children[g-1].children[k-1]
        while header:
            core = core.union(header.value)
            header = header.next

    return core


def querying_for_two_level(tree, k,g):
    max_g = len(tree.children)
    if g > max_g:
        return None
    max_k = len(tree.children[g - 1].children)
    if k > max_k:
        return None
    else:

        starters = []
        header = tree.children[g-1].children[k-1]

        while header:
            starters.append(header)
            header = header.jump

        core = set()
        for s in starters:
            while s:
                core = core.union(s.value)
                s = s.next
    return core

def querying_for_diagonal(tree, k, g):
    max_g = len(tree.children)
    if g > max_g:
        return None
    max_k = len(tree.children[g - 1].children)
    if k > max_k:
        return None
    else:

        starters = []
        header = tree.children[g - 1].children[k - 1]

        while header:
            starters.append(header)
            header = header.jump

        core = set()
        for s in range(len(starters)):
            head = starters[s]
            core = core.union(head.value)
            if s != 0 :
                for i in range(1,s+1):
                    try:
                        core = core.union(head.aux[i])
                    except KeyError:
                        continue
            head = head.next
            cnt = 1
            while head:
                core = core.union(head.value)
                for i in range(1,cnt+1):
                    try :
                        core = core.union(head.aux[i])
                    except KeyError:
                        continue
                head = head.next
                cnt += 1

    return core
#
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Querying Algorithms")
    parser.add_argument("--file_path", help="Path to the Dataset")
    parser.add_argument("--type", type=str, help="Type of Index")
    # parser.add_argument("--k", type=int, help="Value of k")
    # parser.add_argument("--g", type=int, help="Value of g")
    args = parser.parse_args()

    # Load hypergraph

    file_path = args.file_path + "/network.hyp"
    hypergraph, E = load_hypergraph(file_path)

    file_name = args.file_path.split('/')[-1]
    tree_file = f"{file_name}_{args.type}.pkl"

    naive_path = f"{args.file_path}/{file_name}_naive.pkl"
    with open(naive_path, 'rb') as f:
        naive = pickle.load(f)

    values = {}
    for c in naive.children:
        for v in c.children:
            values[v.name] = len(v.value)

    sorted_values = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
    quarter_point = int(len(sorted_values) / 4)
    half_point = int(len(sorted_values) / 2)
    three_quarter_point = int(len(sorted_values) * 3 / 4)

    quarter_k, quarter_g = sorted_values[quarter_point][0]
    half_k, half_g = sorted_values[half_point][0]
    three_quarter_k, three_quarter_g = sorted_values[three_quarter_point][0]
    four_k, four_g = 1, 1

    tree_path = f"{args.file_path}/{tree_file}"
    with open(tree_path, 'rb') as f:
        index_tree = pickle.load(f)
    # with open('/Users/kimsong/real/contact/contact_naive.pkl', 'rb') as f:
    #     index_tree = pickle.load(f)

    start_time = time.time()
    if args.type == 'naive':
        quarter_time_start = time.time()
        core = querying_for_naive_index(index_tree, quarter_k, quarter_g)
        quarter_time_end = time.time()
        half_time_start = time.time()
        core = querying_for_naive_index(index_tree, half_k, half_g)
        half_time_end = time.time()
        three_quarter_time_start = time.time()
        core = querying_for_naive_index(index_tree, three_quarter_k, three_quarter_g)
        three_quarter_time_end = time.time()
        four_time_start = time.time()
        core = querying_for_naive_index(index_tree, four_k, four_g)
        four_time_end = time.time()
    elif args.type == 'one_level':
        quarter_time_start = time.time()
        core = querying_for_one_level(index_tree, quarter_k, quarter_g)
        quarter_time_end = time.time()
        half_time_start = time.time()
        core = querying_for_one_level(index_tree, half_k, half_g)
        half_time_end = time.time()
        three_quarter_time_start = time.time()
        core = querying_for_one_level(index_tree, three_quarter_k, three_quarter_g)
        three_quarter_time_end = time.time()
        four_time_start = time.time()
        core = querying_for_one_level(index_tree, four_k, four_g)
        four_time_end = time.time()
    elif args.type == 'jump':
        quarter_time_start = time.time()
        core = querying_for_two_level(index_tree, quarter_k, quarter_g)
        quarter_time_end = time.time()
        half_time_start = time.time()
        core = querying_for_two_level(index_tree, half_k, half_g)
        half_time_end = time.time()
        three_quarter_time_start = time.time()
        core = querying_for_two_level(index_tree, three_quarter_k, three_quarter_g)
        three_quarter_time_end = time.time()
        four_time_start = time.time()
        core = querying_for_two_level(index_tree, four_k, four_g)
        four_time_end = time.time()
    elif args.type == 'diag':
        quarter_time_start = time.time()
        core = querying_for_diagonal(index_tree, quarter_k, quarter_g)
        quarter_time_end = time.time()
        half_time_start = time.time()
        core = querying_for_diagonal(index_tree, half_k, half_g)
        half_time_end = time.time()
        three_quarter_time_start = time.time()
        core = querying_for_diagonal(index_tree, three_quarter_k, three_quarter_g)
        three_quarter_time_end = time.time()
        four_time_start = time.time()
        core = querying_for_diagonal(index_tree, four_k, four_g)
        four_time_end = time.time()
    end_time = time.time()

    # Write results to file
    output_dir = os.path.dirname(file_path)
    output_filename = f"{args.type}_index_querying.dat"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as output_file:
        # Write number of nodes
        nodes = str(len(core))
        output_file.write("number of nodes: " + nodes + "\n")
        # Write running time
        output_file.write("quarter running time: " + f"{quarter_k,quarter_g} : {quarter_time_end - quarter_time_start}\n")
        output_file.write("half running time: " + f"{half_k,half_g} : {half_time_end- half_time_start}\n")
        output_file.write(
            "three_quarter running time: " + f"{three_quarter_k, three_quarter_g} : {three_quarter_time_end - three_quarter_time_start}\n")
        output_file.write("four running time: " + f"{1,1} : {four_time_end- four_time_start}\n")
        if args.type == 'naive':
            output_file.write("max_g: " + f"{len(index_tree.children)}\n")
            output_file.write("max_k: " + f"{len(index_tree.children[0].children)}\n")
        # Write nodes
        nodes = " ".join(str(node) for node in core)
        output_file.write("nodes in the core: " + nodes + "\n")
    print(f"Results written to {output_path}")