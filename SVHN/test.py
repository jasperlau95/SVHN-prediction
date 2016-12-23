"""
Evaluation for svhn
"""
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('cls', True,
                            """Determine whether target is classification or not."""
                            """ If True, we solve a classification problem,"""
                            """ else a detection problem.""")
tf.app.flags.DEFINE_string('eval_log_dir', './log/classification/eval',
                           """Directory where to write evaluation event logs""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './log/classification/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 26032,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, top_k_predict_op, summary_op, images):
    """
    Run eval once.

    Args:
        saver: Saver/
        summary_writer: Summary writer.
        top_k_op: Top k op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:

        saver.restore(sess, "model.ckpt-30000")
        global_step = "30000"
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            #print (num_iter)
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = float(num_iter * FLAGS.batch_size)
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(top_k_op)
                #image, test_labels = sess.run([images,top_k_predict_op])
                #print (step, int(test_labels[0]))

                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print (total_sample_count)
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """
    Eval svhn for a number of steps.
    """
    with tf.Graph().as_default() as g:
        # Get images and labels for svhn.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = svhn.inputs(eval_data=eval_data)

        # Build a graph that computes the logits predictions from the
        # inference model.
        logits = svhn.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        top_k_predict_op = tf.argmax(logits,1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            svhn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_log_dir, g)

        eval_once(saver, summary_writer, top_k_op, top_k_predict_op, summary_op, images)

def main(argv=None):
    if FLAGS.cls is True:
        # Solve a classification problem
        svhn.cls_extract_test()
    else:
        # Solve a detection
        FLAGS.train_log_dir = './log/detection/eval'
        #svhn.det_extract()
    if tf.gfile.Exists(FLAGS.eval_log_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_log_dir)
    tf.gfile.MakeDirs(FLAGS.eval_log_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
