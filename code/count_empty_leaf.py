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

def count_total_leaf(tree):
    count = 0
    for i in range(len(tree.children)):
        for j in range(len(tree.children[i].children)):
            count +=1

    return count

def sum_of_all_values_in_the_map(map):
    sum = 0
    for key in map:
        sum += map[key]
    return sum

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
    output_filename = "empty_leaf.dat"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as output_file:
        # Write number of nodes
        total_leaf = count_total_leaf(diag)
        output_file.write("number of total leaf nodes : " + f"{total_leaf}\n")
        empty_leaf = count_empty_leaf(diag)
        output_file.write("number of empty leaf nodes with respect to g: " + f"{empty_leaf}\n")

        ratio = sum_of_all_values_in_the_map(empty_leaf) / total_leaf
        output_file.write("ratio of empty leaf nodes out of total leaf node: " + f"{ratio}\n")
    print(f"Results written to {output_path}")