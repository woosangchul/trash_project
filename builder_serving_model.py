import tensorflow as tf

export_path = '\\tmp\\inceptionv4_serving_restavailable_v5\\inceptionV4_v4_10000\\1'

print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

prediction_classes = ['Glass','Plastic']
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

with tf.Graph().as_default() as g:
    with tf.Session() as new_sess:
        with tf.gfile.FastGFile("inception_v4_graph_rest_available.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        
        input_placeholder = new_sess.graph.get_tensor_by_name("image_preprocessing/input_bytes:0")
        softmax = new_sess.graph.get_tensor_by_name("InceptionV4/Logits/Predictions:0")
        processed_image = new_sess.graph.get_tensor_by_name("image_preprocessing/Reshape_1:0")
        
        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_placeholder)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(softmax)
        tensor_info_image = tf.saved_model.utils.build_tensor_info(processed_image)
        prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y,
                'images' : tensor_info_image
                },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
        new_sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
        'predict_images':
            prediction_signature
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

builder.save()

""" tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    classification_signature,"""