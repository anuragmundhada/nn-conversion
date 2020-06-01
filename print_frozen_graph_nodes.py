# -*- coding: utf-8 -*-
# @Time    : 18-7-16 上午10:26
# @Author  : edvard_hua@live.com
# @FileName: gen_tflite_coreml.py
# @Software: PyCharm

import argparse
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description="Tools for convert frozen_pb into tflite or coreml.")
parser.add_argument("--frozen_pb", type=str, default="./hourglass/model-360000.pb", help="Path for storing checkpoint.")

args = parser.parse_args()

# output_filename = args.frozen_pb.rsplit("/", 1)[1]
# output_filename = output_filename.split(".")[0]


def get_graph_def_from_file(graph_filepath):
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def



def optimize_graph(graph_def, transforms, output_node):
  input_names = []
  output_names = [output_node]
  optimized_graph_def = TransformGraph(graph_def, input_names,      
      output_names, transforms)
  tf.train.write_graph(optimized_graph_def, logdir=model_dir, as_text=False, 
     name='optimized_model.pb')
  print('Graph optimized!')




output_filename ='model'
tf.compat.v1.reset_default_graph()
graph_def = tf.compat.v1.GraphDef()
with tf.compat.v1.Session() as sess:
    # Read binary pb graph from file
    with tf.compat.v1.gfile.Open(args.frozen_pb, "rb") as f:
        data2read = f.read()
        graph_def.ParseFromString(data2read)

    tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

    # Get Nodes

    all_nodes = sess.graph.get_operations()
    conv_nodes = [n for n in sess.graph.get_operations() if n.type in ['Conv2D','MaxPool','AvgPool', 'DepthwiseConv2dNative', 'Resize']]
    
    for node in all_nodes:
        atts = {key:node.node_def.attr[key] for key in list(node.node_def.attr.keys()) if key != 'data_format'}
        
        print(node.name, node.type, sess.graph.get_tensor_by_name(node.name+':0').shape)
        
    print("\nNum conv nodes: ", len(conv_nodes), "All nodes: ", len(all_nodes))




