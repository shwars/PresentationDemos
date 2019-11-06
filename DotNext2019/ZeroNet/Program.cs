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


namespace ZeroNet
{

    class Program
    {
 
        static void Main(string[] args)
        {
            Console.WriteLine("Execution begins...");

            NetByHand.MainFunc(args);
            ProgramTF.MainFunc(args);
            PKeras.MainFunc(args);
            MLNET.MainFunc(args);
            ProgramTrans.MainFunc(args);

            Console.WriteLine("Thanks for all the fish");
        }
    }
}
