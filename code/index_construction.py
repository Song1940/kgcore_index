import networkx as nx
import time
import os
import argparse
import copy
from types import MappingProxyType
import pickle
from pympler import asizeof
import operator
import sys

sys.setrecursionlimit(10000)

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

"""hypergraph에 속한 각각 노드에 대해 이웃 노드들과, 이웃 노드들과의 co-occurence를 반환하는 함수"""
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



def get_induced_subhypergraph(hypergraph, node_set):
    subhypergraph = nx.Graph()
    for node in node_set:
        if node in hypergraph.nodes:
            subhypergraph.add_node(node, hyperedges=[])
            for hyperedge in hypergraph.nodes[node]['hyperedges']:
                p = node_set & set(hyperedge)
                if len(p) >= 2:
                    subhypergraph.add_edges_from([(u, v) for u in p for v in p if u != v])
                    subhypergraph.nodes[node]['hyperedges'].append(p)
    return subhypergraph

""" (k,g)-core peeling algorithm"""
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



def kg_core_peeling(hypergraph,k,g):
    changed = True
    H = set(hypergraph.nodes())
    T = set()
    while changed:
        changed = False
        if T == set():
            nodes = H.copy()
        else:
            nodes = H.intersection(T)
        for v in nodes:
            T = T - {v}
            map = neighbour_count_map(hypergraph,v,g)
            map =  {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
            if len(map) < k:
                changed = True
                H -= {v}
                T.union(get_neighbour(hypergraph,v))
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
# def enumerate_kg_core_fixing_k(hypergraph, E, k):
#     H = set(hypergraph.nodes)
#     S = []
#     for g in range(1,len(E)):
#         if len(H) <= k:
#             break
#         while True:
#             if len(H) <= k:
#                 break
#             changed = False
#             nodes = H.copy()
#             for v in nodes:
#                 map = neighbour_count_map(hypergraph,v,g)
#                 map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
#                 if len(map) < k:
#                     H -= {v}
#                     changed = True
#             if not changed and len(H) != 0:
#                 S.append(H.copy())
#                 break
#
#     return S





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




def enumerate_1_g(hypergraph, g):
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

# def enumerate_1_k(hypergraph,E, k):
#
#     H = set(hypergraph.nodes)
#     S = []
#     temp = set()
#     for g in range(1,len(E)):
#         if len(H) <= k:
#             break
#         while True:
#             if len(H) <= k:
#                 break
#             changed = False
#             nodes = H.copy()
#             for v in nodes:
#                 map = neighbour_count_map(hypergraph,v,g)
#                 map = {neighbor: count for neighbor, count in map.items() if neighbor in nodes}
#                 if len(map) < k:
#                     H -= {v}
#                     changed = True
#             if not changed:
#                 if len(temp) != 0:
#                     S.append(temp.difference(H))
#                 temp = H.copy()
#                 break
#     if len(temp) != 0:
#         S.append(temp)
#
#     return S


def one_level_compression(hypergraph, E):
    T = TreeNode("root")
    for g in range(1, len(E)):
        S = enumerate_1_g(hypergraph, g)
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

def jump_compression(hypergraph,E):
    h_time_start = time.time()
    T_1 = one_level_compression(hypergraph,E)
    h_time_end = time.time()
    T_2 = T_1
    max_g = len(T_2.children)
    for g in range(max_g-1):
        max_k = len(T_2.children[g+1].children)
        for k in range(max_k):
            T_2.children[g].children[k].jump = T_2.children[g+1].children[k]
            T_2.children[g].children[k].value = T_2.children[g].children[k].value.difference(T_2.children[g].children[k].jump.value)

    h_time = h_time_end - h_time_start
    return T_2, h_time

def diagonal_compression(hypergraph,E):
    v_time_start = time.time()
    T,h_time = jump_compression(hypergraph,E)
    v_time_end = time.time()
    for g in range(len(T.children)):
        head = T.children[g].children[0] ## 1,1 에서 시작
        if head.next == None:
            break
        for k in range(len(T.children[g].children)-1):
            if head.next and head.jump : ## 1,2 , 2,1 있으면 2,1을 head 로 시작
                diag = head.jump
                head = head.next
                intersect = head.value.intersection(diag.value) ## 2,1의 value와 1,2의 value의 교집합

                if not diag.next:
                    head.jump = TreeNode("aux") ## temp.jump == diag.next
                    diag.next = head.jump

                diag.next.aux[1] = intersect
                for i in diag.aux.keys():
                    try :
                        diag.next.aux[i+1] = diag.aux[i].intersection(head.aux[i])
                        head.aux[i] = head.aux[i].difference(diag.next.aux[i + 1])
                    except KeyError:
                        continue

                head.value = head.value.difference(intersect)
                if head.next and head.next.aux:
                    head.value = head.value.difference(head.next.aux[1])

            else: ## 없으면 head.next에서 발생한 중복 처리
                while head.next:
                    for i in head.next.aux.keys():
                        if i ==1:
                            head.value = head.value.difference(head.next.aux[i])
                        else:
                            head.aux[i-1] = head.aux[i-1].difference(head.next.aux[i])
                    head = head.next

    v_time = v_time_end - v_time_start
    ### 마지막 g값의 k값들에 대해서도 중복처리해야함

    return T, h_time, v_time

def count_total_nodes(tree, type):
    if type == "naive":
        total = 0
        for i in range(len(tree.children)):
            for j in range(len(tree.children[i].children)):
                total += len(tree.children[i].children[j].value)
        return total

    total = 0
    if type == 'diag':
        for i in range(len(tree.children)):
            head = tree.children[i].children[0]
            while head:
                for j in head.aux.keys():
                    total += len(head.aux[j])
                total += len(head.value)
                head = head.next
        return total
    else:
        for i in range(len(tree.children)):
            head = tree.children[i].children[0]
            while head:
                total += len(head.value)
                head = head.next
        return total

def count_each_nodes(tree, type):
    count_map =dict()
    if type == 'diag':
        for i in range(len(tree.children)):
            head = tree.children[i].children[0]
            while head:
                for j in head.aux.keys():
                    for k in head.aux[j]:
                        try:
                            count_map[k] += 1
                        except KeyError:
                            count_map[k] = 1

                for j in head.value:
                    try:
                        count_map[j] += 1
                    except KeyError:
                        count_map[j] = 1
                head = head.next

        count_map = dict(sorted(count_map.items(), key=operator.itemgetter(1), reverse=True))
        return count_map

    else:
        for i in range(len(tree.children)):
            for j in range(len(tree.children[i].children)):
                for k in tree.children[i].children[j].value:
                    try:
                        count_map[k] += 1
                    except KeyError:
                        count_map[k] = 1

        count_map = dict(sorted(count_map.items(), key=operator.itemgetter(1), reverse=True))
        return count_map

def count_empty_leaf(tree):
    map = {}
    for i in range(len(tree.children)):
        count = 0
        head = tree.children[i].children[0]
        while head:
            if (len(head.value) + len(head.aux)) == 0:
                count += 1
            head = head.next

        map[i+1]  = count
    return map



#

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="(k,g)-core Index Construction")
    parser.add_argument("--file_path", help="Path to the network file")
    parser.add_argument("--type", type=str, help="Type of Index")
    args = parser.parse_args()

    # Load hypergraph
    hypergraph, E = load_hypergraph(args.file_path)

    # Measure the running time
    start_time = time.time()
    if args.type == 'naive':
        index_tree = naive_index_construction(hypergraph, E)
    elif args.type == 'one_level':
        index_tree = one_level_compression(hypergraph,E)
    elif args.type == 'jump':
        index_tree,h_time = jump_compression(hypergraph,E)
    elif args.type == 'diag':
        index_tree,h_time,v_time = diagonal_compression(hypergraph,E)
    end_time = time.time()

    # Write results to file
    output_dir = os.path.dirname(args.file_path)
    output_filename = f"{args.type}_index.dat"
    output_path = os.path.join(output_dir, output_filename)


    with open(output_path, 'w') as output_file:
        # Write number of nodes
        nodes = str(count_total_nodes(index_tree, args.type))
        output_file.write("number of nodes: " + nodes + "\n")
        # Write running time
        output_file.write("running time: " + f"{end_time - start_time}\n")
        if args.type =='diag':
            output_file.write("horizontal_compression_time: " + f"{h_time}\n")
            output_file.write("vertical_compression_time: " + f"{v_time-h_time}\n")
            total_time = end_time - start_time
            output_file.write("diagonal_compression_time: " + f"{total_time-(v_time-h_time)-h_time}\n")
        # Write memory usage of the index tree
        memory_usage = asizeof.asizeof(index_tree)
        output_file.write("memory usage: " + str(memory_usage) + "\n")
        # Write number of nodes in each level
        node_count = count_each_nodes(index_tree, args.type)
        output_file.write("frequency of each nodes : " + f"{node_count}\n")
        # Write number of empty leaf nodes
        empty_leaf = count_empty_leaf(index_tree)
        output_file.write("number of empty leaf nodes with respect to g: " + f"{empty_leaf}\n")

    # saving index tree in output path
    file_name = args.file_path.split('/')[-2]
    with open(f'{output_dir}/{str(file_name)+"_"+(args.type)}.pkl', 'wb') as f:
        pickle.dump(index_tree, f)

    # Write memory usage of the index tree

    print(f"Results written to {output_path}")
#
# with open('/Users/kimsong/real/contact/contact_naive.pkl', 'rb') as f:
#     index_tree = pickle.load(f)