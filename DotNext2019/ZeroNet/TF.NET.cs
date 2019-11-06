using Microsoft.ML.Transforms;
using NumSharp;
using System;
using System.IO;
using System.Linq;
using System.Net.WebSockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using Tensorflow;
using Tensorflow.Estimators;
using static Tensorflow.Binding;
using static ZeroNet.UtilFuncs;

namespace ZeroNet
{
    class ProgramTF
    {

        public static void MainFunc(string[] args)
        {
            Console.WriteLine("Execution begins...");

            var Rnd = new Random();

            Console.WriteLine("Loading data...");
            var train =
                File.ReadAllLines(@"c:\data\mnist\train.csv")
                .Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray())
                .Take(10000)
                .ToArray();
            var test =
                File.ReadAllLines(@"c:\data\mnist\train.csv")
                .Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray())
                .Skip(10000)
                .Take(500)
                .ToArray();

            var graph = new Graph().as_default();
            var x = tf.placeholder(tf.float32, shape: (-1, 784), name: "X");
            var y = tf.placeholder(tf.float32, shape: (-1, 10), name: "Y");

            var initer = tf.truncated_normal_initializer(stddev: 0.01f);
            var W1 = tf.get_variable("W1", (784, 100), tf.float32, initer);
            var b1 = tf.get_variable("b1", 100, tf.float32);
            var z1 = tf.matmul(x, W1) + b1;
            var zr = tf.nn.relu(z1);

            var W2 = tf.get_variable("W2", (100, 10), tf.float32, initer);
            var b2 = tf.get_variable("b2", 10, tf.float32);
            var z = tf.matmul(zr, W2) + b2;

            var logits = tf.nn.softmax_cross_entropy_with_logits(y, z);
            var loss = tf.reduce_mean(logits, name: "loss");

            var optimizer = tf.train.AdamOptimizer(learning_rate: 0.001f, name: "Adam-op").minimize(loss);

            var correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1), name: "correct_pred");
            var accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name: "accuracy");
            var cls_prediction = tf.argmax(z, axis: 1, name: "predictions");

            Console.WriteLine("One-hot encoding...");
            var train_labels = (from t in train
                                select OH((int)t[0])).ToArray();
            var test_labels = (from t in test
                                select OH((int)t[0])).ToArray();

            Console.WriteLine("Normalizing...");
            var train_norm = train.Select(z => z.Select(x => x / 255.0).ToArray()).ToArray();
            var test_norm = test.Select(z => z.Select(x => x / 255.0).ToArray()).ToArray();

            var batch_size = 32;
            NDArray xt, yt;
            Console.WriteLine("Training...");
            float loss_v, acc_v;
            using (var sess = tf.Session())
            {
                sess.run(tf.global_variables_initializer());
                for (var ep = 0; ep < 7; ep++)
                {
                    var n = 0;
                    while (n + batch_size < train.Length)
                    {
                        xt = new NDArray(train_norm[n..(n + batch_size)])[":,1:"];
                        yt = new NDArray(train_labels[n..(n + batch_size)]);
                        sess.run(optimizer, (x, xt), (y, yt));
                        n += batch_size;
                    }
                    xt = new NDArray(test_norm)[":,1:"];
                    yt = new NDArray(test_labels);
                    (loss_v, acc_v) = sess.run((loss, accuracy), (x, xt), (y, yt));
                    Console.WriteLine($"Epoch: {ep} -- Training Loss={loss_v}, Acc={acc_v}");
                }
            }

            Console.WriteLine("Thanks for all the fish");
            Console.ReadKey();
        }
    }
}
