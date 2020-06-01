# generic imports
import cv2
import tqdm
import os
import numpy as np
import h5py
import copy
import time
import argparse
import pdb
import glob

# conversion imports
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # stop printing all the tensorflow information-level logs
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import onnxruntime


def convert_nchw_to_nhwc_all(graph_def_file, output_file):
    tf.compat.v1.reset_default_graph()
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.Session() as sess:
        # Read binary pb graph from file
        with tf.compat.v1.gfile.Open(graph_def_file, "rb") as f:
            data2read = f.read()
            graph_def.ParseFromString(data2read)
        tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

        all_nodes = sess.graph.get_operations()
        conv_nodes = [n for n in sess.graph.get_operations() if n.type in ['Conv2D','MaxPool','AvgPool', 'DepthwiseConv2dNative', 'Resize']]
        print("NCHW_TO_NHWC: All nodes", len(all_nodes), "| Conv nodes: ", len(conv_nodes))
        # for node in all_nodes:            
        for idx, n_org in enumerate(all_nodes):
            if n_org.type == 'Const':
                pass
            elif n_org.type == 'Placeholder':
                pass
            elif n_org.type == 'Transpose':
                org_inp_tens = sess.graph.get_tensor_by_name(n_org.inputs[0].name)
                op_inputs = [org_inp_tens]
                atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
                atts = {}
                op = sess.graph.create_op(op_type='Identity', inputs=op_inputs, name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts)
                # Update Connections
                out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
                out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
                for out in out_nodes:
                    for j, nam in enumerate(out.inputs):
                        if n_org.outputs[0] == nam:
                            try:
                                out._update_input(j, out_tens)
                            except:
                                import pdb; pdb.set_trace()

                import pdb; pdb.set_trace()
                # atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
                # op = sess.graph.create_op(op_type='Transpose', inputs=[], name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts)
                # out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
                # out_trans = tf.transpose(out_tens, [0, 2, 3, 1], name=n_org.name +'_transp_out')
                # # Update Connections
                # op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts)
                # out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
                # for out in out_nodes:
                #     for j, nam in enumerate(out.inputs):
                #         if n_org.outputs[0] == nam:
                #             out._update_input(j, out_trans)

            elif n_org.type in ['Conv2D','MaxPool','AvgPool', 'DepthwiseConv2dNative']:
                assert len(n_org.inputs)==1 or len(n_org.inputs)==2
                org_inp_tens = sess.graph.get_tensor_by_name(n_org.inputs[0].name)
                # inp_tens = tf.compat.v1.transpose(org_inp_tens, [0, 2, 3, 1], name=n_org.name +'_transp_input')
                op_inputs = [org_inp_tens]
                
                # Get filters for Conv but don't transpose
                if n_org.type == 'Conv2D':
                    filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
                    op_inputs.append(filter_tens)

                if n_org.type == 'DepthwiseConv2dNative':
                    filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
                    op_inputs.append(filter_tens)

                # Attributes without data_format, NWHC is default
                atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
                if n_org.type in ['MaxPool', 'AvgPool']:
                    kl = atts['ksize'].list.i
                    ksl = [kl[0], kl[2], kl[3], kl[1]]
                    st = atts['strides'].list.i
                    stl = [st[0], st[2], st[3], st[1]]
                    atts['ksize'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=ksl))
                    atts['strides'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=stl))

                if n_org.type == 'Conv2D':
                    st = atts['strides'].list.i
                    stl = [st[0], st[2], st[3], st[1]]
                    atts['strides'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=stl))

                if n_org.type == 'DepthwiseConv2dNative':
                    st = atts['strides'].list.i
                    stl = [st[0], st[2], st[3], st[1]]
                    atts['strides'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=stl))

                # Create new Operation
                #print(n_org.type, n_org.name, list(n_org.inputs), n_org.node_def.attr['data_format'])
                # op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', attrs=atts) 
                # if n_org.type == 'DepthwiseConv2dNative':
                #     import pdb; pdb.set_trace()
                op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts)

                out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
                # out_trans = tf.transpose(out_tens, [0, 3, 1, 2], name=n_org.name +'_transp_out')
                # if n_org.type == 'DepthwiseConv2dNative' or n_org.type == 'Conv2D':
                #     import pdb; pdb.set_trace()

                # try:
                #     assert out_trans.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape
                # except:
                #     import pdb; pdb.set_trace()
                #     print("######################### FIXME: Somehow the tensor shape comparison failed")
                
                # Update Connections
                out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
                for out in out_nodes:
                    for j, nam in enumerate(out.inputs):
                        if n_org.outputs[0] == nam:
                            out._update_input(j, out_tens)
            else:
                print(n_org.type)
                import pdb; pdb.set_trace()




def convert_nchw_to_nhwc(graph_def_file, output_file):

    tf.compat.v1.reset_default_graph()
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.Session() as sess:
        # Read binary pb graph from file
        with tf.compat.v1.gfile.Open(graph_def_file, "rb") as f:
            data2read = f.read()
            graph_def.ParseFromString(data2read)
        tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

        # Get Nodes

        all_nodes = sess.graph.get_operations()
        conv_nodes = [n for n in sess.graph.get_operations() if n.type in ['Conv2D','MaxPool','AvgPool', 'DepthwiseConv2dNative', 'Resize']]
        print("NCHW_TO_NHWC: All nodes", len(all_nodes), "| Conv nodes: ", len(conv_nodes))
        # for node in all_nodes:            
        #     print(node.name, node.type, sess.graph.get_tensor_by_name(node.name+':0').shape)

        for idx, n_org in enumerate(all_nodes):
            if n_org.type in ['Conv2D','MaxPool','AvgPool', 'DepthwiseConv2dNative']:
                # Transpose input
                assert len(n_org.inputs)==1 or len(n_org.inputs)==2
                org_inp_tens = sess.graph.get_tensor_by_name(n_org.inputs[0].name)
                inp_tens = tf.compat.v1.transpose(org_inp_tens, [0, 2, 3, 1], name=n_org.name +'_transp_input')
                op_inputs = [inp_tens]
                
                # Get filters for Conv but don't transpose
                if n_org.type == 'Conv2D':
                    filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
                    op_inputs.append(filter_tens)

                if n_org.type == 'DepthwiseConv2dNative':
                    filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
                    op_inputs.append(filter_tens)

                # Attributes without data_format, NWHC is default
                atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
                if n_org.type in ['MaxPool', 'AvgPool']:
                    kl = atts['ksize'].list.i
                    ksl = [kl[0], kl[2], kl[3], kl[1]]
                    st = atts['strides'].list.i
                    stl = [st[0], st[2], st[3], st[1]]
                    atts['ksize'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=ksl))
                    atts['strides'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=stl))

                if n_org.type == 'Conv2D':
                    st = atts['strides'].list.i
                    stl = [st[0], st[2], st[3], st[1]]
                    atts['strides'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=stl))

                if n_org.type == 'DepthwiseConv2dNative':
                    st = atts['strides'].list.i
                    stl = [st[0], st[2], st[3], st[1]]
                    atts['strides'] = tf.compat.v1.AttrValue(list=tf.compat.v1.AttrValue.ListValue(i=stl))

                # Create new Operation
                #print(n_org.type, n_org.name, list(n_org.inputs), n_org.node_def.attr['data_format'])
                # op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', attrs=atts) 
                # if n_org.type == 'DepthwiseConv2dNative':
                #     import pdb; pdb.set_trace()
                op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts)

                out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
                out_trans = tf.transpose(out_tens, [0, 3, 1, 2], name=n_org.name +'_transp_out')
                # if n_org.type == 'DepthwiseConv2dNative' or n_org.type == 'Conv2D':
                #     import pdb; pdb.set_trace()

                try:
                    assert out_trans.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape
                except:
                    import pdb; pdb.set_trace()
                    print("######################### FIXME: Somehow the tensor shape comparison failed")
                
                # Update Connections
                out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
                for out in out_nodes:
                    for j, nam in enumerate(out.inputs):
                        if n_org.outputs[0] == nam:
                            out._update_input(j, out_trans)
            elif n_org.type in ['Const', 'Placeholder', 'Identity', 'AddV2', 'Minimum', 'Min', 'Mul',
                                'Maximum', 'Reshape', 'Split', 'Transpose', 'Sub', 'Pad', 'ConcatV2',
                                'Rsqrt', 'Max', 'Add', 'ExpandDims', 'FusedBatchNormV3']:
                pass
            else:
                print("NEW OPERATION in NHWC conversion: ", n_org.type)
            
        # Delete old nodes
        graph_def = sess.graph.as_graph_def()
        for on in conv_nodes:
            graph_def.node.remove(on.node_def)

        # Write graph
        tf.compat.v1.io.write_graph(graph_def, "", output_file, as_text=False)


def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():

    model_name = 'mobilenetv2_coco_2020-05-07-15-26'
    model_name = 'mobilenetv2_coco_2020-05-14-12-29'
    model_name = 'mobilenetv2_coco_2020-05-14-12-57'
    model_name = 'mobilenetv2_coco_2020-05-14-13-14'
    model_name = 'mobilenetv2_coco_2020-05-14-13-22'
    model_name = 'mobilenetv2_coco_2020-05-14-13-52'
    model_name = 'mobilenetv2_coco_2020-05-14-14-30'
    model_name = 'mobilenetv2_coco_2020-05-14-15-02'
    model_name = 'mobilenetv2_coco_2020-05-15-13-00'
    # model_name = 'mobilenetv2_coco_2020-05-15-15-14'
    model_name = 'latest'
    # model_name = 'mobilenetv2_coco_2020-05-21-21-44'

    RES = 192

    if model_name == 'latest':
        experiments = glob.glob('/home/anurag/workspace/fast-human-pose-estimation.pytorch/output/mpii/pose_mobilenetv2/*')
        experiments.sort()
        model_name = experiments[-1].split('/')[-1] 


    print("Converting model {}".format(model_name))

    if not os.path.exists('conversion_out/{}'.format(model_name)):
        os.mkdir('conversion_out/{}'.format(model_name))

    # Sample data
    
    dude = cv2.imread('/home/anurag/workspace/fast-human-pose-estimation.pytorch/dude.webp')
    resized = cv2.resize(dude, (RES, RES) , interpolation = cv2.INTER_LINEAR)
    X_test = resized.reshape((1, RES, RES, 3)).astype(np.float32)/255
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # # # Step 0: load pytorch model & weights
    import _init_paths
    import models
    from config import cfg, update_config

    args = parse_args()
    update_config(cfg, args)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    checkpoint_file = '/home/anurag/workspace/fast-human-pose-estimation.pytorch/output/mpii/pose_mobilenetv2/{}/model_best.pth'.format(model_name)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    dummy_input = torch.from_numpy(X_test.copy()).cuda()
    dummy_output_pytorch = model(dummy_input).cpu().data.numpy()

    print("Pytorch model loaded. Input shape: ", dummy_input.shape, ". Output shape: ", dummy_output_pytorch.shape)

    ## # # # # #  Step 1: Convert to ONNX # # # # # # 
    """
    Note:
    opset 11 is required for correct conversion of resize / upsample layers.
    However, it is not yet available in onnx-tensorflow yet. It is being tracked at
    https://github.com/onnx/onnx-tensorflow/issues/449
    https://github.com/pytorch/pytorch/issues/34718
    """
    # dummy_input_for_onnx = torch.autograd.Variable(torch.randn(1, 3, RES, RES)).cuda() # nchw
    # torch.onnx.export(model, dummy_input, 'conversion_out/{}/model.onnx'.format(model_name),
    #                   input_names=['input'], output_names=['output'],
    #                   opset_version=10)
    # print("pytorch model exported to ONNX")

    # # # Step 2: Load ONNX model and convert to TensorFlow
    # model_onnx = onnx.load('conversion_out/{}/model.onnx'.format(model_name))
    # onnx.checker.check_model(model_onnx)

    # # # print model in human readable format
    # onnx.helper.printable_graph(model_onnx.graph)

    # tf_rep = prepare(model_onnx)
    # print("onnx model converted to tensorflow")
    # #### #### Print out tensors and placeholders in model (helpful during inference in TensorFlow)

    # #### #### Step 3: Export model as .pb file (frozen TF graph)
    # tf_rep.export_graph('conversion_out/{}/model.pb'.format(model_name))
    # print("model saved as frozen graph")


    # # # # # # # # Step 4: Convert to NHWC  # # # # # # 
    convert_nchw_to_nhwc_all('conversion_out/{}/model_optim.pb'.format(model_name), 'conversion_out/{}/model_nhwc.pb'.format(model_name))
    print("model saved as NHWC")


    # # # # # # Step 5: Convert to tf lite # # # # # # 
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('conversion_out/{}/model.pb'.format(model_name),
                                                          input_arrays=['input'], # input arrays 
                                                          output_arrays=['output'],  # output arrays as told in upper in my model case it si add_10
                                                          input_shapes={'input' :[1, 3, RES, RES]}
    )

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # Not sure what this does 
    # converter.experimental_new_converter = False
    # ## # ## using this option gives higher conversion error (~1e-3) while timing on oneplus is the same
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantize = False
    if quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        dataset_list = tf.data.Dataset.list_files("/home/anurag/datasets/mpii/images/*.jpg")

        def representative_data_gen():
          for input_value in dataset_list.take(100):
            img = tf.io.read_file(input_value)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [RES, RES])
            img = tf.transpose(img, perm=[2,0,1])
            img = tf.expand_dims(img, axis=0)
            yield [img]

        converter.representative_dataset = representative_data_gen

    # convert the model 
    tf_lite_model = converter.convert()
    # save the converted model 
    open('conversion_out/{}/model_optim.tflite'.format(model_name), 'wb').write(tf_lite_model)
    print("model saved as tflite")







    # Step 6: Testing
    do_test = True
    if do_test:
        ## # # # # #  Step 6-1: ONNX output # # # # # # 
        ort_session = onnxruntime.InferenceSession('conversion_out/{}/model.onnx'.format(model_name))

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: X_test}
        dummy_output_onnx = ort_session.run(None, ort_inputs)[0]

        error_pytorch2onnx = np.mean(np.abs(dummy_output_pytorch - dummy_output_onnx))
        # np.testing.assert_allclose(dummy_output_pytorch, dummy_output_onnx, rtol=1e-03, atol=1e-05)

        # # # # # # # Step 6-2: NCHW Tensorflow output  # # # # # # 
        graph = tf.Graph()
        sess = tf.compat.v1.InteractiveSession(graph = graph)

        with tf.compat.v1.gfile.GFile('conversion_out/{}/model.pb'.format(model_name), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        input_placeholder = tf.compat.v1.placeholder(np.float32, shape = [1, 3, RES, RES], name='input')
        tf.compat.v1.import_graph_def(graph_def, {'input': input_placeholder})

        layers = [op.name for op in graph.get_operations()]
        output_tensor = graph.get_tensor_by_name("import/output:0")
        dummy_output_tensorflow_orig = sess.run(output_tensor, feed_dict = {input_placeholder: X_test})

        sess.close()

        # # # # # # # Step 6-3: NHWC Tensorflow output  # # # # # # 
        graph = tf.compat.v1.Graph()
        sess = tf.compat.v1.InteractiveSession(graph = graph)

        with tf.compat.v1.gfile.GFile('conversion_out/{}/model_nhwc.pb'.format(model_name), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        input_placeholder = tf.compat.v1.placeholder(np.float32, shape = [1, 3, RES, RES], name='input')
        tf.compat.v1.import_graph_def(graph_def, {'input': input_placeholder})

        layers = [op.name for op in graph.get_operations()]
        output_tensor = graph.get_tensor_by_name("import/output:0")
        dummy_output_tensorflow = sess.run(output_tensor, feed_dict = {input_placeholder: X_test})

        sess.close()

        # # # # # # # Step 6-4: TFLite output  # # # # # # 

        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path='conversion_out/{}/model.tflite'.format(model_name))
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], X_test)
        interpreter.invoke()
        dummy_output_tflite = interpreter.get_tensor(output_details[0]['index'])


        # # # # # # # Stepwise error  # # # # # # 

        error_pytorch2onnx = np.mean(np.abs(dummy_output_pytorch - dummy_output_onnx))
        error_onnx2tforig = np.mean(np.abs(dummy_output_onnx - dummy_output_tensorflow_orig))
        error_tforig2tfnhwc = np.mean(np.abs(dummy_output_tensorflow_orig - dummy_output_tensorflow))
        error_tfnhwc2tflite = np.mean(np.abs(dummy_output_tensorflow - dummy_output_tflite))

        error_pytorch2tflite = np.mean(np.abs(dummy_output_pytorch - dummy_output_tflite))

        print("pytorch2onnx error:", error_pytorch2onnx)
        print("onnx2tforig error:", error_onnx2tforig)
        print("tforig2tfnhwc error:", error_tforig2tfnhwc)
        print("tfnhwc2tflite error", error_tfnhwc2tflite)
        print("Final error -> pytorch2tflite error:", error_pytorch2tflite)


        # # Visualize heatmaps - left is pytorch, right is tflite
        # for idx in range(16):
        #     pytorch_hm = dummy_output_pytorch[0][idx]
        #     onnx_hm = dummy_output_onnx[0][idx]
        #     tensorflow_hm = dummy_output_tensorflow[0][idx]
        #     tflite_hm = dummy_output_tflite[0][idx]
        #     combined_image = np.hstack((pytorch_hm, onnx_hm, tensorflow_hm, tflite_hm))
        #     cv2.imshow('d', combined_image)
        #     cv2.waitKey()


if __name__ == '__main__':
    main()
