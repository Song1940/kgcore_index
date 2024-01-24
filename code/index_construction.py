import networkx as nx
import time
import os
import argparse
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
def neighbour_count_map(hypergraph, v,g):
    neighbor_counts = {}
    for hyperedge in hypergraph.nodes[v]['hyperedges']:
        # Increment the count for each neighbor in the hyperedge
        for neighbor in hyperedge:
            if neighbor != v:
                neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

    filtered_neighbors = {neighbor: count for neighbor, count in neighbor_counts.items() if count >= g}


    return filtered_neighbors


def get_neighbour(hypergraph, v):
    neighbor_set= set()
    for hyperedge in hypergraph.nodes[v]['hyperedges']:
        # Increment the count for each neighbor in the hyperedge
        for neighbor in hyperedge:
            if neighbor != v:
                neighbor_set.add(neighbor)

    return neighbor_set



""" updated (k,g)-core peeling algorithm"""
def find_kg_core(hypergraph,k,g):
    changed = True
    H = set(hypergraph.nodes)
    while changed:
        changed = False
        nodes = H.copy()
        for v in nodes:
            map = neighbour_count_map(hypergraph,v,g)
            map =  {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
            if len(map) < k:
                changed = True
                H -= {v}
    return H




def enumerate_kg_core_fixing_g(hypergraph, g):
    H = set(hypergraph.nodes)
    S = []
    T = set()
    for k in range(1,len(hypergraph.nodes)):
        if len(H) <= k:
            break
        while True:
            if len(H) <= k:
                break
            changed = False
            nodes = H.copy()
            if T == set():
                for v in nodes:
                    T = T - {v}
                    map = neighbour_count_map(hypergraph,v,g)
                    map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
                    if len(map) < k:
                        H -= {v}
                        changed = True
                        T.union(get_neighbour(hypergraph,v))
            else:
                for v in nodes.intersection(T):
                    T = T - {v}
                    map = neighbour_count_map(hypergraph,v,g)
                    map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
                    if len(map) < k:
                        H -= {v}
                        changed = True
                        T.union(get_neighbour(hypergraph,v))
            if not changed:
                S.append(H.copy())
                T = set()
                break

    return S

""" naive index construction"""
def naive_index_construction(hypergraph,E):
    T = TreeNode("root")
    for g in range(1,len(E)):
        S = enumerate_kg_core_fixing_g(hypergraph,g)
        if len(S) == 0:
            break
        T.children.append(TreeNode(g))
        for s in range(len(S)):
            T.children[g-1].children.append(TreeNode((s+1,g)))
            T.children[g - 1].children[s].value = S[s]

    return T




def enumerate_h(hypergraph, g):
    H = set(hypergraph.nodes)
    S = []
    T = set()
    temp = set()
    for k in range(1,len(hypergraph.nodes)):
        if len(H) <= k:
            break
        while True:
            if len(H) <= k:
                break
            changed = False
            nodes = H.copy()
            if T != set():
                nodes = nodes.intersection(T)
            for v in nodes:
                T = T - {v}
                map = neighbour_count_map(hypergraph,v,g)
                map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
                if len(map) < k:
                    H -= {v}
                    changed = True
            if not changed:
                if len(temp) != 0:
                    S.append(temp.difference(H))
                    T = set()
                temp = H.copy()
                break
    if len(temp) != 0:
        S.append(temp)


    return S



def horizontal_compression(hypergraph, E):
    T = TreeNode("root")
    for g in range(1, len(E)):
        S = enumerate_h(hypergraph, g)
        if len(S) == 0:
            break
        T.children.append(TreeNode(g))
        prev = TreeNode(None)
        for s in range(len(S)):
            T.children[g - 1].children.append(TreeNode((s + 1, g)))
            T.children[g - 1].children[s].value = S[s]
            u = T.children[g-1].children[s]
            if prev.name != None:
                prev.next = u
            prev = u

    return T

def vertical_compression(hypergraph,E):
    T_h = horizontal_compression(hypergraph,E)
    T_v = T_h
    max_g = len(T_v.children)
    for g in range(max_g-1):
        max_k = len(T_v.children[g+1].children)
        for k in range(max_k):
            T_v.children[g].children[k].jump = T_v.children[g+1].children[k]
            T_v.children[g].children[k].value = T_v.children[g].children[k].value.difference(T_v.children[g].children[k].jump.value)

    return T_v

def diagonal_compression(hypergraph,E):
    T = vertical_compression(hypergraph,E)
    for g in range(len(T.children)):
        head = T.children[g].children[0]
        if head.next == None:
            break
        for k in range(len(T.children[g].children)-1):
            if head.next and head.jump :
                diag = head.jump
                head = head.next
                intersect = head.value.intersection(diag.value)

                if not diag.next:
                    head.jump = TreeNode("aux")
                    diag.next = head.jump

                diag.next.aux[1] = intersect

                for i in diag.aux.keys():
                    try :
                        if diag.aux[i].intersection(head.aux[i]):
                            diag.next.aux[i+1] = diag.aux[i].intersection(head.aux[i])
                            head.aux[i] = head.aux[i].difference(diag.next.aux[i + 1])
                    except KeyError:
                        continue

                head.value = head.value.difference(intersect)
                if head.next and head.next.aux:
                    head.value = head.value.difference(head.next.aux[1])

            else:
                while head.next:
                    for i in head.next.aux.keys():
                        if i ==1:
                            head.value = head.value.difference(head.next.aux[i])
                        else:
                            head.aux[i-1] = head.aux[i-1].difference(head.next.aux[i])
                    head = head.next
    return T





if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="(k,g)-core Index Construction")
    parser.add_argument("--file_path", help="Path to the network file")
    parser.add_argument("--type", type=str, help="Type of Index")
    args = parser.parse_args()

    # Load hypergraph
    hypergraph, E = load_hypergraph(args.file_path)

    # Index construction
    start_time = time.time()
    if args.type == 'naive':
        index_tree = naive_index_construction(hypergraph, E)
    elif args.type == 'hori':
        index_tree = horizontal_compression(hypergraph,E)
    elif args.type == 'vert':
        index_tree,h_time = vertical_compression(hypergraph,E)
    elif args.type == 'diag':
        index_tree,h_time,v_time = diagonal_compression(hypergraph,E)


    # Write results to file
    output_dir = os.path.dirname(args.file_path)
    output_filename = f"{args.type}_index.dat"
    output_path = os.path.join(output_dir, output_filename)


    # Saving index tree in output path
    file_name = args.file_path.split('/')[-2]
    with open(f'{output_dir}/{str(file_name)+"_"+(args.type)}.pkl', 'wb') as f:
        pickle.dump(index_tree, f)


    print(f"Results written to {output_path}")

