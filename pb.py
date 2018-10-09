#python freeze_graph.py --input_graph output/graph.pbtxt  --input_checkpoint ./baybayin-v2-0.0001-4convlayers.model --output_graph /tmp/out --output_node_names sample_name

import tensorflow as tf
saver = tf.train.import_meta_graph('baybayin-v2-0.0001-4convlayers.model.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./baybayin-v2-0.0001-4convlayers.model")