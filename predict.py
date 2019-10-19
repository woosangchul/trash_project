import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys



# 11.3.2
tf.app.flags.DEFINE_string("output_graph", "/example/trash_graph.pb", "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels", "/example/trash_labels.txt", "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image", True, "이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS

def main(_):

    # 11.3.3
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]

    # 11.3.4
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    # 11.3.5
    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')

        # 11.3.6
        image = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})  

    # 11.3.7
    print('=== 예측 결과 ===')
    print(labels)

    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print(' %s (%.2f%%)' % (name, score * 100))

    # 11.3.8
    if FLAGS.show_image:
        img = mpimg.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()

# 11.3.9
if __name__ == "__main__":
    tf.app.run()