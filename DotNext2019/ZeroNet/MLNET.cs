using Microsoft.ML;
using Microsoft.ML.Data;
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


namespace ZeroNet
{
    public class MLDigit
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(1,784)]
        [VectorType(784)]
        public float[] Pixels { get; set; }
    }

    class MLNET
    {

        public static void MainFunc(string[] args)
        {
            Console.WriteLine("Execution begins...");

            Console.WriteLine("Loading data...");
            var ctx = new MLContext();

            var data = ctx.Data.LoadFromTextFile<MLDigit>(@"c:\data\mnist\train.csv",',',true);
            var pipe = ctx.Transforms.Conversion.MapValueToKey("Number", "Label")
                .Append(ctx.Transforms.NormalizeMinMax("Pixels"));
            
            var trans = pipe.Fit(data);
            var newdata = trans.Transform(data);

            var split = ctx.Data.TrainTestSplit(newdata);
            
            Console.WriteLine("Training model...");
            var model = ctx.MulticlassClassification
                           .Trainers.NaiveBayes(featureColumnName: "Pixels", labelColumnName: "Number");
            var trained_model = model.Fit(split.TrainSet);
            var predictions = trained_model.Transform(split.TestSet);
            var metrics = ctx.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
            Console.WriteLine(metrics.MacroAccuracy);

            Console.WriteLine("Thanks for all the fish");
            Console.ReadKey();
        }
    }
}
