using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Utils;
using Python.Runtime;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Numpy;
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

    class ProgramTrans
    {
        public class Pet
        {
            public string Path { get; set; }
            public bool Label { get; set; }
        }


        public static void MainFunc(string[] args)
        {
            Console.WriteLine("Execution begins...");

            Console.WriteLine("Loading data...");
            var dir = @"c:\data\catsdogs";
            var data = Directory.GetFiles(dir)
                .Select(f => new Pet()
                {
                    Path  = f,
                    Label = f.Contains("cat.") //? "cat" : "dog"
                });

            var ctx = new MLContext();

            var split = ctx.Data.TrainTestSplit(ctx.Data.LoadFromEnumerable(data),0.8);

            var tfm = @"c:\data\models\inception\tensorflow_inception_graph.pb";

            var pipe = ctx.Transforms.LoadImages("Image", @"c:\data\catsdogs", "Path")
                .Append(ctx.Transforms.ResizeImages("ImageResized", 244, 244, "Image"))
                .Append(ctx.Transforms.ExtractPixels("input", "ImageResized", interleavePixelColors: true, offsetImage: 117))
                .Append(ctx.Model.LoadTensorFlowModel(tfm).ScoreTensorFlowModel("softmax1_pre_activation", "input", true))
                .Append(ctx.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "softmax1_pre_activation"));

            Console.WriteLine("Training model...");
            var model = pipe.Fit(split.TrainSet);

            Console.WriteLine("Testing model...");
            var test = model.Transform(split.TestSet);
            var metrics = ctx.BinaryClassification.Evaluate(test, "Label");

            Console.WriteLine(metrics.Accuracy);

            Console.WriteLine("Thanks for all the fish");
            Console.ReadKey();
        }
    }
}
