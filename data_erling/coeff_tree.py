import numpy as np

class Node:

    def __init__(self, value=None, parent=None, children=None):
        if parent==None:
            self.level=0
        else:
            self.level = parent.level + 1
        self.value = value
        self.parent = parent
        self.children = []
        self.children.append(children)
        self.coeff_value = 1.0  #Is the average for the area price node, and the coefficients for the leafs

    def set_coeff_value(self, value):
        self.coeff_value = value
    

    def set_parent(self, parent):
        if self.parent != parent:
            self.parent = parent
            parent.set_child(self)

    def set_child(self, child):
        if child not in self.children:
            self.children.append(child)
            child.set_parent(self)

    def get_child(self, child):
        if child in self.children:
            return child
        else:
            print('Not a child_node')

    def get_children(self):
        if len(self.children) == 0:
            print('No children for this node')
        else:
            return self.children

    def print_subtree(self):
        for child in self.children:
            child.print_subtree()
        print(self.value)