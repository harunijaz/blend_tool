from ShapeBlender import *

if __name__ == "__main__":

    # Test data
    #source_obj_file = "data/CB1_CB2/Source/meshes/Seat.obj"
    #target_obj_file = "data/CB1_CB2/Target/meshes/Seat.obj"
    #num_samples = 10
    blend = ShapeBlender()

    # Load source and target shapes
    #source_shape = blend.load_shape_from_obj(source_obj_file, "source")
    #target_shape = blend.load_shape_from_obj(target_obj_file, "target")

    # Parse Source XML files
    path_to_source_shape_xml = "data/CB1_CB2/Source/SimpleChair1.xml"
    source_shape_graph = blend.create_graph_from_xml(path_to_source_shape_xml)
    #print(f"Source data: {len(list(source_shape_graph.edges))}")
    # Parse Target XML file
    path_to_target_shape_xml = "data/CB1_CB2/Target/shortChair01.xml"
    target_shape_graph = blend.create_graph_from_xml(path_to_target_shape_xml)
    #print(f"Target data: {len(list(target_shape_graph.edges))}")

    correspondence_txt = blend.parse_correspondence_txt_file("data/CB1_CB2/correspondence.txt")
    #print(correspondence_txt)

    # Create augmented graph
    created_augmented_graph = blend.create_augmented_graph(source_shape_graph, target_shape_graph, correspondence_txt)
    #print(list(created_augmented_graph.edges))

    #blend.add_edges_to_augmented_graph(created_augmented_graph, source_shape_graph)
    """
    # Sample blending paths and generate in-between shapes
    blending_paths = blend.stochastically_sample_blending_paths(G, G0, num_samples)
    inbetween_shapes = []
    for blending_path in blending_paths:
        inbetween_shape = blend.generate_inbetween_shape(G, G0, blending_path)
        inbetween_shapes.append(inbetween_shape)

    # Apply implausibility filter
    plausible_shapes = blend.implausibility_filter(inbetween_shapes, source_shape, target_shape)

    # Print the number of plausible shapes
    print("Number of plausible shapes: ", len(plausible_shapes)) """


