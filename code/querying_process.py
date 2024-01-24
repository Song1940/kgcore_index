import networkx as nx
import os
import argparse
import pickle


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

def querying_for_horizontal(tree, k,g):
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


def querying_for_vertical(tree, k,g):
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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Querying Algorithms")
    parser.add_argument("--file_path", help="Path to the Dataset")
    parser.add_argument("--type", type=str, help="Type of Index")
    parser.add_argument("--k", type=int, help="Value of k")
    parser.add_argument("--g", type=int, help="Value of g")
    args = parser.parse_args()


    # Load index
    tree_file = f"{}.pkl"
    tree_path = f"{args.file_path}/{tree_file}"
    with open(tree_path, 'rb') as f:
        index_tree = pickle.load(f)


    if args.type == 'naive':
        core = querying_for_naive_index(index_tree, args.k, args.g)
    elif args.type == 'hori':
        core = querying_for_horizontal(index_tree, args.k, args.g)
    elif args.type == 'vert':
        core = querying_for_vertical(index_tree, args.k, args.g)
    elif args.type == 'diag':
        core = querying_for_diagonal(index_tree, args.k, args.g)

    # Write results to file
    output_dir = os.path.dirname(args.file_path)
    output_filename = f"{args.type}_index_querying.dat"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as output_file:
        # Write number of nodes
        nodes = str(len(core))
        output_file.write("number of nodes: " + nodes + "\n")

        # Write nodes
        nodes = " ".join(str(node) for node in core)
        output_file.write(f"nodes in the ({args.k,args.g})-core: " + nodes + "\n")
    print(f"Results written to {output_path}")