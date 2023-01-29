import numpy as np
from lxml import etree
import networkx as nx
import csv

class ShapeBlender:

    def __init__(self):
        self.source_shape = None
        self.target_shape = None
        self.correspondence = None
        self.source_graph = None
        self.target_graph = None
        self.augmented_graph = None

    def create_augmented_graph(self, graph_source, graph_target, correspondence):
        # Create a copy of graph_source
        self.augmented_graph = graph_source.copy()
        '''
        here we are doing this for source graph so remember that there may be one-to-many from either 
        source shape or target shape, so rows order matter in correspondence.txt file.
        we need to update this code here while preparing more corresponding data. most of code is manual.
        in future, we will handle if more than one records on one-to-many relations
        '''
        target_nodes_new = []
        for value in correspondence.values():
            if len(value) == 1:
                source_node_old = value
            if len(value) > 1:
                for v in value:
                    target_nodes_new.append(v)
        self.replace_node_edges_augmented_graph(source_node_old, np.array(target_nodes_new))
        print(f"Length: {len(list(self.augmented_graph.nodes))}")

    def replace_node_edges_augmented_graph(self, old_node, new_nodes):
        # Get the edges connected to the old node and also Convert type networkx_edge into list
        edges = list(self.augmented_graph.edges(old_node))

        # Remove the old node from the graph
        self.augmented_graph.remove_node(old_node[0])

        # Add the new nodes to the graph
        for new_node in new_nodes:
            self.augmented_graph.add_node(new_node)

        # Add edges to the new nodes
        for edge in edges:
            for new_node in new_nodes:
                self.augmented_graph.add_edge(edge[1], new_node)

    def stochastically_sample_blending_paths(self, G, G0, num_paths):
        blending_paths = []
        for i in range(num_paths):
            current_G = G.copy()
            path = []
            while current_G != G0:
                node = random.choice(list(current_G.nodes))
                corresponding_node = self.find_corresponding_node(node, G0)
                if corresponding_node:
                    task = self.determine_blending_task(node, corresponding_node)
                    path.append((node, task))
                    current_G = self.execute_task(current_G, task)
            blending_paths.append(path)
        return blending_paths

    '''
    def implausibility_filter(self, G, G0):
        if not self.check_global_reflection_symmetry(G, G0):
            return False
        if not self.check_part_group_symmetry(G, G0):
            return False
        if not self.check_part_connection(G, G0):
            return False
        return True

    def check_global_reflection_symmetry(self, G, G0):
        # Check if both G and G0 have global reflection symmetry
        if not has_global_reflection_symmetry(G) or not has_global_reflection_symmetry(G0):
            return True
        deviation = global_reflection_symmetry_deviation(G)
        bounding_box_diagonal = average_bounding_box_diagonal(G, G0)
        if deviation > 0.05 * bounding_box_diagonal:
            return False
        return True

    def check_part_group_symmetry(self, G, G0):
        # Check if all present group symmetries in G are preserved
        for group in find_part_groups(G0):
            if not group_symmetry_preserved(group, G):
                return False
        return True

    def check_part_connection(self, G, G0):
        # Check if all connected parts in G0 are still connected in G
        for part1, part2 in find_connected_parts(G0):
            if not parts_still_connected(part1, part2, G):
                return False
        return True

    def grow(self, null_node, G0):
        seed_part = self.find_seed_part(null_node, G0)
        if seed_part:
            # Interpolate shape of null_node along path of natural continuation
            self.interpolate_shape(null_node, seed_part)
            self.interpolate_position(null_node, seed_part)
            symmetry_group = self.perform_split(null_node, G, G0)
            if symmetry_group:
                self.relink_to_symmetry(null_node, symmetry_group, G)
            else:
                self.relink_attachment(null_node, G)
        else:
            print("No seed part found for null node: ", null_node)
        return G

    def shrink(self, null_node, G0):
        # Find corresponding node in G0
        corresponding_node = self.find_corresponding_node(null_node, G0)
        # Find seed part in G0
        seed_part = self.find_seed_part(null_node, G0)

        # Interpolate position of null node along path of natural continuation
        self.interpolate_position(null_node, seed_part, corresponding_node)

        # Perform split if necessary
        self.perform_split(null_node, G0)

        # Relink to symmetry group if necessary
        self.relink_to_symmetry(null_node, corresponding_node.symmetry_group)

        # Interpolate shape of null node
        self.interpolate_shape(null_node, corresponding_node)

        # Relink attachment of null node
        self.relink_attachment(null_node)

        # Remove null node from G^
        G.remove_node(null_node)

        # Return G^
        return G

    def morph(self, nodes, target_graph):
        for node in nodes:
            corresponding_node = self.find_corresponding_node(node, target_graph)
            if corresponding_node:
                # Interpolate position of contact points along the curves/sheets 
                # connecting the start and end positions
                self.interpolate_position(node, corresponding_node)
                # Interpolate shape of node towards corresponding_node
                self.interpolate_shape(node, corresponding_node)
            else:
                # Handle case where corresponding node is not found
                pass

    
    def interpolate_shape(node, corresponding_node):
        # Find corresponding points on source and target shapes
        source_point = node.position
        target_point = corresponding_node.position

        # Set up local orthonormal frame
        tangent, normal, binormal = find_local_frame(node)

        # Interpolate local encodings
        interpolated_tangent = interpolate(source_point.tangent, target_point.tangent)
        interpolated_normal = interpolate(source_point.normal, target_point.normal)
        interpolated_binormal = interpolate(source_point.binormal, target_point.binormal)

        # Compute surface point corresponding to in-between node
        new_position = interpolated_tangent + interpolated_normal + interpolated_binormal
        node.position = new_position

    
    def find_local_frame(point):
        # Code to calculate tangent, normal, and binormal at point
        tangent = calculate_tangent(point)
        normal = calculate_normal(point)
        binormal = calculate_binormal(point)
        return tangent, normal, binormal
    '''

    def interpolate(source, target):

        #source = [1, 2, 3]
        #target = [4, 5, 6]
        #in_between = interpolate(source, target)
        #print(in_between) # [2.5, 3.5, 4.5]
        alpha = 0.5
        in_between = source + (target - source) * alpha
        return in_between

    '''
    def find_seed_part(self, null_node, G0):
        # Get the corresponding null node in the target shape
        corresponding_null_node = self.find_corresponding_node(null_node, G0)
        # Get the connected parts of the corresponding null node in the target shape
        corresponding_parts = list(nx.neighbors(G0, corresponding_null_node))
        # Randomly select one of the connected parts as the seed part
        seed_part = random.choice(corresponding_parts)
        return seed_part

    def perform_split(self, node, G, G0):
        corresponding_node = self.find_corresponding_node(node, G0)
        split_nodes = []
        if corresponding_node:
            # Find the seed part for the split
            seed_part = self. (node, corresponding_node)
            # Perform split
            split_parts = self.split_part(seed_part)
            # Create new nodes in G for each split part
            for split_part in split_parts:
                new_node = Node(split_part)
                G.add_node(new_node)
                split_nodes.append(new_node)
        return split_nodes
    
    '''
    def relink_to_symmetry(self, node, symmetry_group, G):

        # find the corresponding parts in the symmetry group
        corresponding_parts = [self.find_corresponding_node(part, G) for part in symmetry_group]
        # filter out any None values
        corresponding_parts = [part for part in corresponding_parts if part]
        if len(corresponding_parts) == 1:
            # if the node is only connected to one part, move it to the contact point
            contact_point = self.find_contact_point(node, corresponding_parts[0])
            node.position = contact_point
        elif len(corresponding_parts) == 2:
            # if the node is connected to two parts, move it to the midpoint of the contact points
            contact_point1 = self.find_contact_point(node, corresponding_parts[0])
            contact_point2 = self.find_contact_point(node, corresponding_parts[1])
            node.position = (contact_point1 + contact_point2) / 2
        elif len(corresponding_parts) > 2:
            # if the node is connected to more than two parts, move it to the centroid of the contact points
            contact_points = [self.find_contact_point(node, part) for part in corresponding_parts]
            node.position = sum(contact_points) / len(contact_points)
    '''
    def relink_attachment(node, G):
        contact_points = node['attachment']
        if len(contact_points) == 1:
            # Find the connected part and translate the node to the contact point
            for connected_part in G.neighbors(node):
                node['position'] = contact_points[0]
                break
        elif len(contact_points) == 2:
            # Find the connected parts and transform the node to attach at the contact points
            connected_parts = list(G.neighbors(node))
            transformation = calculate_transformation(node['position'], contact_points[0], contact_points[1],
                                                      connected_parts[0]['position'],
                                                      connected_parts[1]['position'])
            node['position'] = transform_point(node['position'], transformation)
            node['shape'] = transform_shape(node['shape'], transformation)
        elif len(contact_points) > 2:
            # Consider the node as rigid and translate it to the centroid of the contact points
            centroid = calculate_centroid(contact_points)
            node['position'] = centroid
        '''

    def calculate_transformation(source_points, target_points):
        '''
        This function takes in two input parameters, source_points and target_points,
        which are the corresponding points on the source and target shapes, respectively.
        It first calculates the centroid of the source and target points using numpy's mean function.
        Then it shifts the points to the origin by subtracting the centroid from each point.
        Next, it calculates the covariance matrix using numpy's dot function.
        It then calculates the singular value decomposition of the covariance matrix
        using numpy's svd function. From the SVD, it calculates the rotation matrix and
        the translation vector, which are used to transform the source shape to match the target shape.
        '''
        source_points = np.array(source_points)
        target_points = np.array(target_points)
        # calculate centroid of source and target points
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        # shift points to origin
        source_points -= source_centroid
        target_points -= target_centroid
        # calculate the covariance matrix
        cov_matrix = np.dot(source_points.T, target_points)
        # calculate the singular value decomposition
        U, S, V_T = np.linalg.svd(cov_matrix)
        # calculate the rotation matrix
        R = np.dot(U, V_T)
        # calculate the translation vector
        t = target_centroid - np.dot(R, source_centroid)
        return R, t

    def transform_point(point, transformation_matrix):
        '''
        This function takes in a 3D point and a transformation matrix,
        and applies the transformation to the point by performing a matrix
        multiplication of the point's homogeneous coordinates with the transformation matrix.
        The function returns the transformed point in 3D coordinates.
        '''
        point_homo = np.append(point, 1)
        transformed_point_homo = np.matmul(transformation_matrix, point_homo)
        transformed_point = transformed_point_homo[:3]
        return transformed_point

    def transform_shape(shape, transformation):
        """
        Transforms the shape using the given transformation matrix
        :param shape: the shape to be transformed (list of points)
        :param transformation: the transformation matrix (3x3 or 4x4)
        :return: the transformed shape (list of points)
        """
        transformed_shape = []
        for point in shape:
            transformed_point = np.dot(transformation, point)
            transformed_shape.append(transformed_point)
        return transformed_shape

    def calculate_centroid(points):
        '''
        This function will take a list of points in the form of [x, y, z]
        and return the centroid (average of x, y, and z coordinates) as a list in the same form.
        '''
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        centroid = [sum(x_coords) / len(points), sum(y_coords) / len(points), sum(z_coords) / len(points)]
        return centroid

    def create_graph_from_xml(self, file_path):
        tree = etree.parse(file_path)
        root = tree.getroot()
        # Create a new empty graph
        new_graph = nx.Graph()
        # Iterate over the nodes in the XML file
        for node in root.findall("node"):
            node_id = node.find("id").text
            new_graph.add_node(node_id)
        # Iterate over the edges in the XML file
        for edge in root.findall("edge"):
            n = edge.findall('n')
            new_graph.add_edge(n[0].text, n[1].text)
        return new_graph

    def load_shape_from_obj(self, file_path, shape_type):
        """
        load_shape_from_obj method that takes in two parameters: file_path and shape_type.
        The method reads an obj file at the specified file path and loads the data into
        a dictionary containing two keys "vertices" and "faces", representing the vertex
        and face data of the shape respectively. The shape_type parameter is used to
        determine whether the loaded shape should be stored as the source, target, or inbetween shape.
        """
        try:
            vertices = []
            faces = []
            with open(file_path) as f:
                for line in f:
                    if line[0] == "v":
                        vertex = list(map(float, line[2:].strip().split()))
                        vertices.append(vertex)
                    elif line[0] == "f":
                        face = list(map(int, line[2:].strip().split()))
                        faces.append(face)

            shape_data = {"vertices": vertices, "faces": faces}


            if shape_type == "source":
                self.source_shape = shape_data
            elif shape_type == "target":
                self.target_shape = shape_data
            elif shape_type == "inbetween":
                self.inbetween_shape = shape_data
            else:
                raise ValueError(
                    "Invalid shape type: {}. Must be one of 'source', 'target', or 'inbetween'.".format(shape_type))

            return shape_data

        except FileNotFoundError:
            print(f"{file_path} not found.")
        except:
            print("An error occurred while loading the shape.")

    def parse_correspondence_txt_file(self, file_path):
        correspondences = {}
        line_number = 0
        with open(file_path, 'r') as file:
            for line in file:
                line_number += 1
                if line_number > 1:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        source_name = parts[0]
                        target_names = []
                        for i in range(1, len(parts)):
                            target_names.append(parts[i])
                        correspondences[source_name] = target_names
        return correspondences


