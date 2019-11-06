using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Utils;
using Python.Runtime;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Numpy;
// using NumSharp;
using System;
using System.Collections.Generic;
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

    class NetByHand
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
                .Where(x => x[0] < 1.1)
                .Take(3000)
                .ToArray();
            var test =
                File.ReadAllLines(@"c:\data\mnist\train.csv")
                .Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray())
                .Where(x => x[0] < 1.1)
                .Skip(3000)
                .Take(500)
                .ToArray();

            var W = train[0][1..].Select(_ => Rnd.NextDouble() * 2.0 - 1.0).ToArray();

            Console.WriteLine("Training...");
            foreach (var t in train)
            {
                var res = Mult(t[1..], W);
                if (t[0] < 0.1 && res < 0)
                {
                    W = Add(W, t[1..]);
                }
                if (t[0] > 0.9 && res > 0)
                {
                    W = Sub(W, t[1..]);
                }
            }

            int n = 0, c = 0;
            foreach (var t in test)
            {
                if (t[0] > 1.01) continue;
                var res = Mult(t[1..], W);
                if ((res >= 0.0 && Math.Abs(t[0] - 0.0) < 0.01) ||
                    (res < 0.0 && Math.Abs(t[0] - 1.0) < 0.01)) c++;
                n++;
            }
            Console.WriteLine("Total: {0}, Correct: {1}, Accuracy: {2}", n, c, (double)c / (double)n);

            Console.WriteLine("Thanks for all the fish");
        }
    }
}
