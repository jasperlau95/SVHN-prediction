import tensorflow as tf
import svhn
svhn.cls_extract_test()
images = svhn.inputs_test('test')
logits = svhn.inference(images)
top_k_predict_op = tf.argmax(logits,1)
saver = tf.train.Saver(tf.all_variables())
with tf.Session() as sess:
    saver.restore(sess, "model.ckpt-150000")
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    step = 0
    fp = open("labels.txt", "w")
    while step < 1000 and not coord.should_stop():
        image, test_labels = sess.run([images,top_k_predict_op])
        if int(test_labels[0]) == 0:
            fp.write("10\n")
        else:
            fp.write("%d\n" %test_labels[0])
        step += 1
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=120)