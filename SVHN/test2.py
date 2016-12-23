import numpy as np
import tensorflow as tf
import svhn
import math
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_examples', 26032,
                            """Number of examples to run.""")

#def main(argv=None):
svhn.cls_extract_test()
#evaluate()
# Get images and labels for svhn.
images, labels = svhn.inputs('test')

# inference model.
logits = svhn.inference(images)

# Calculate predictions.
top_k_op = tf.nn.in_top_k(logits, labels, 1)
top_k_predict_op = tf.argmax(logits,1)

# Restore the moving average version of the learned variables for eval.
variable_averages = tf.train.ExponentialMovingAverage(svhn.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt-29999")
    global_step = "29999"
    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True, start=True))

    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    print (num_iter)
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = float(num_iter * FLAGS.batch_size)
    step = 0
    #fp = open("labels.txt", "w")
    while step < num_iter and not coord.should_stop():
        predictions = sess.run(top_k_op)
        image, test_labels = sess.run([images,top_k_predict_op])
        #if int(test_labels[0]) == 0:
        #    fp.write("10\n")
        #else:
        #    fp.write("%d\n" %test_labels[0])

        #if step % 200 == 0:
        print (step, int(test_labels[0]))
        true_count += np.sum(predictions)
        step += 1

    #fp.close()
    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print (total_sample_count)
    print('precision @ 1 = %.3f' %precision)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=120)

#if __name__ == '__main__':
#    tf.app.run()
