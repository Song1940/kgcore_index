import argparse
import os
import pickle
import time
import operator

class TreeNode:
    def __init__(self, name):
        self.aux ={}
        self.name = name
        self.children = []
        self.next = None
        self.value = set()
        self.jump = None

def aux_node_count(diag):
    map = {}
    for i in range(len(diag.children)):
        head = diag.children[i].children[0]
        while head:
            if head.aux:
                for j in head.aux.keys():
                    if len(head.aux[j]) != 0:
                        try:
                            map[j] += 1
                        except:
                            map[j] = 1
            head = head.next
    return map


def num_aux_node_depth(diag):
    map = {}
    for i in range(len(diag.children)):
        head = diag.children[i].children[0]
        while head:
            if head.aux:
                for j in head.aux.keys():
                    if len(head.aux[j]) != 0:
                        try:
                            map[j] += len(head.aux[j])
                        except:
                            map[j] = len(head.aux[j])
            head = head.next
    return map


def num_aux_node(diag):
    count = 0
    for i in range(len(diag.children)):
        head = diag.children[i].children[0]
        while head:
            if head.aux:
                if head.aux[1]:
                    count += 1
            head = head.next
    return count

def avg_aux_node(diag):
    total = 0
    for i in range(len(diag.children)):
        head = diag.children[i].children[0]
        while head:
            if head.aux:
                for j in head.aux.keys():
                    total += len(head.aux[j])
            head = head.next

    return total

def total_node(diag):
    count = 0
    for i in range(len(diag.children)):
        head = diag.children[i].children[0]
        while head:
            count += len(head.value)
            head = head.next
    return count


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Querying Algorithms")
    parser.add_argument("--file_path", help="Path to the Dataset")
    args = parser.parse_args()

    # Load hypergraph
    file_path = args.file_path + "/network.hyp"


    file_name = args.file_path.split('/')[-1]
    tree_file = f"{file_name}_diag.pkl"

    diag_path = f"{args.file_path}/{file_name}_diag.pkl"
    with open(diag_path, 'rb') as f:
        diag = pickle.load(f)




    # Write results to file
    output_dir = os.path.dirname(file_path)
    output_filename = "aux_node.dat"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as output_file:
        # Write number of aux nodes
        aux_map = aux_node_count(diag)
        num_aux_map = num_aux_node_depth(diag)
        num_of_aux_node = num_aux_node(diag)
        # sorted_map = sorted(aux_map.items(), key=operator.itemgetter(1), reverse=True)
        output_file.write("aux node count: " + f"{aux_map}\n")
        output_file.write("num aux node: " + f"{num_aux_map}\n")
        output_file.write("avg aux node: " + f"{avg_aux_node(diag)/num_of_aux_node}\n")
        output_file.write("total node: " + f"{total_node(diag)}\n")
        output_file.write("num of aux node: " + f"{num_of_aux_node}\n")

    print(f"Results written to {output_path}")