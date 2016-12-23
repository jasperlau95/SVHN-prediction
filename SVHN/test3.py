import numpy as np
import tensorflow as tf
import svhn

svhn.cls_extract_test()
images, labels = svhn.inputs('test')

# inference model.
logits = svhn.inference(images)
top_k_predict_op = tf.argmax(logits,1)

# Restore the moving average version of the learned variables for eval.
variable_averages = tf.train.ExponentialMovingAverage(svhn.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt-30000")
    global_step = "30000"
    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True, start=True))

    step = 0
    fp = open("labels.txt", "w")
    while step < 26032 and not coord.should_stop():
        #predictions = sess.run(top_k_op)
        image, test_labels = sess.run([images,top_k_predict_op])
        if int(test_labels[0]) == 0:
            fp.write("10\n")
        else:
            fp.write("%d\n" %test_labels[0])

        if step % 200 == 0:
            print (step, int(test_labels[0]))

        step += 1

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
