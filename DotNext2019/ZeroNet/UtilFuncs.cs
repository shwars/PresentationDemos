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
using NumSharp;

namespace ZeroNet
{

    public static class UtilFuncs
    {

        public static double[] flatten(double[][] a)
        {
            var L = new List<double>();
            foreach (var x in a)
            {
                L.AddRange(x);
            }
            return L.ToArray();
        }

        public static double[] OH(int n)
        {
            var res = new double[10];
            for (int i = 0; i < 10; i++)
            {
                res[i] = Math.Abs(i - n) < 0.1 ? 1.0 : 0.0;
            }
            return res;
        }
        public static double Mult(double[] a, double[] b)
        {
            return a.Zip(b).Select(x => x.First * x.Second).Sum();
        }

        public static double[] Sub(double[] a, double[] b)
        {
            return a.Zip(b).Select(x => x.First - x.Second).ToArray();
        }
        public static double[] Add(double[] a, double[] b)
        {
            return a.Zip(b).Select(x => x.First + x.Second).ToArray();
        }

        public static double[] Mult(double x, double[] a)
        {
            return a.Select(z => z * x).ToArray();
        }

        public static T[] SkipOne<T>(T[] x)
        {
            return x.Skip(1).ToArray();
        }


  
    }
}
