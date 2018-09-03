using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningBench
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string ModelPath => Path.Combine(AppPath, "SentimentModel.zip");

        static void Main(string[] args)
        {
            //var appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
            //var trainDataPath = Path.Combine(appPath, "sentiment-imdb-train.txt");


            //var pipeline = new LearningPipeline();
            //var textLoader = new TextLoader(trainDataPath).CreateFrom<SentimentData>();

            //pipeline.Add(textLoader);
            //pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            TrainAsync().GetAwaiter().GetResult();

            Console.ReadLine();
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.  
            var pipeline = new LearningPipeline();

            var trainPath = @"C:\Users\maxim\Desktop\aclImdb\train\imdbTraining.txt";

            // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
            // all the column names and their types.
            //pipeline.Add(new TextLoader(TrainDataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextLoader(trainPath).CreateFrom<SentimentData>());

            // TextFeaturizer is a transform that will be used to featurize an input column to format and clean the data.
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            // FastTreeBinaryClassifier is an algorithm that will be used to train the model.
            // It has three hyperparameters for tuning decision tree performance. 
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<SentimentData, SentimentPrediction>();

            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        public class SentimentData
        {
            [Column("0")]
            public string SentimentText;

            [Column("1", name: "Label")]
            public float Sentiment;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment;
        }
    }
}
