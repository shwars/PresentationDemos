﻿using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;

namespace ZeroNet
{
    static class NetByHand
    {
        static double Mult(double[] a, double[] b)
        {
            return a.Zip(b).Select(x => x.First * x.Second).Sum();
        }

        static double[] Sub(double[] a, double[] b)
        {
            return a.Zip(b).Select(x => x.First - x.Second).ToArray();
        }
        static double[] Add(double[] a, double[] b)
        {
            return a.Zip(b).Select(x => x.First + x.Second).ToArray();
        }


        static double[] Mult(double x, double [] a)
        {
            return a.Select(z => z * x).ToArray();
        }

        static T[] SkipOne<T>(T[] x)
        {
            return x.Skip(1).ToArray();
        }

        public static void MainFunc(string[] args)
        {
            Console.WriteLine("Execution begins...");

            var Rnd = new Random();
            Console.WriteLine("Loading data...");
            var train =
                File.ReadAllLines(@"c:\data\mnist\train.csv")
                .Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray())
                .Where(x => x[0]<1.1)
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

            var W = SkipOne(train[0]).Select(_ => Rnd.NextDouble() * 2.0 - 1.0).ToArray();

            Console.WriteLine("Training...");
            foreach (var t in train)
            {
                var res = Mult(SkipOne(t), W);
                if (t[0]<0.1 && res<0)
                {
                    W = Add(W, SkipOne(t));
                }
                if (t[0] > 0.9 && res > 0)
                {
                    W = Sub(W, SkipOne(t));
                }
            }

            int n = 0, c = 0;
            foreach(var t in test)
            {
                if (t[0] > 1.01) continue;
                var res = Mult(SkipOne(t), W);
                if ((res >= 0.0 && Math.Abs(t[0] - 0.0) < 0.01) ||
                    (res < 0.0 && Math.Abs(t[0] - 1.0) < 0.01)) c++;
                n++;
            }
            Console.WriteLine("Total: {0}, Correct: {1}, Accuracy: {2}", n, c, (double)c / (double)n);

            Console.WriteLine("Thanks for all the fish");
            Console.ReadKey();
        }
    }
}
