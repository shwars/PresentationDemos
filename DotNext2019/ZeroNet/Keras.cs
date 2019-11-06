using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Utils;
using Python.Runtime;
using static ZeroNet.UtilFuncs;
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

    public class PKeras
    { 
        public static void MainFunc(string[] args)
        {
            Console.WriteLine("Execution begins...");

            Console.WriteLine("Loading data...");
            var train =
                File.ReadAllLines(@"c:\data\mnist\train.csv")
                .Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray())
                .Take(3000)
                .ToArray();
            var test =
                File.ReadAllLines(@"c:\data\mnist\train.csv")
                .Skip(1)
                .Select(x => x.Split(',').Select(double.Parse).ToArray())
                .Skip(3000)
                .Take(500)
                .ToArray();

            string envPythonHome = "C:\\winapp\\Miniconda3\\envs\\py36\\";
            string envPythonLib = envPythonHome + "Lib;" + envPythonHome + "Lib\\site-packages";
            Environment.SetEnvironmentVariable("PATH", @"c:\winapp\Miniconda3\envs\py36;c:\winapp\Miniconda3\envs\py36\Library\mingw-w64\bin;c:\winapp\Miniconda3\envs\py36\Library\usr\bin;c:\winapp\Miniconda3\envs\py36\Library\bin;c:\winapp\Miniconda3\envs\py36\Scripts;c:\winapp\Miniconda3\envs\py36\bin;c:\winapp\Miniconda3\condabin;c:\winapp\miniconda\bin;c:\winapp\miniconda\scripts", EnvironmentVariableTarget.Process);
            PythonEngine.PythonHome = envPythonHome;

            Console.WriteLine("One-hot encoding...");
            var train_labels = Util.ToCategorical(np.array(train.Select(x => (int)x[0]).ToArray()));
            var test_labels = Util.ToCategorical(np.array(test.Select(x => (int)x[0]).ToArray()));

            Console.WriteLine("Normalizing...");
            var train_norm = np.array(flatten(train)).reshape(-1,785)[":,1:"] / 255.0;
            var test_norm = np.array(flatten(test)).reshape(-1, 785)[":,1:"] / 255.0;

            var model = new Sequential();
            model.Add(new Dense(100, 784, activation: "relu"));
            model.Add(new Dense(10, activation: "softmax"));
            model.Compile(loss: "categorical_crossentropy",
              optimizer: new Adadelta(), metrics: new string[] { "accuracy" });
            model.Fit(train_norm, train_labels, 
                batch_size: 32,
                epochs: 5,
                validation_data: new NDarray[] { test_norm, test_labels });

            Console.WriteLine("Thanks for all the fish");
            Console.ReadKey();
        }
    }
}
