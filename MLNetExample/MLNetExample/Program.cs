using System;
using Microsoft.ML;
using MLNetExample.Model;

namespace MLNetExample
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            //Load data
            var trainData = context.Data.LoadFromTextFile<SalaryData>("Data//SalaryData.csv", hasHeader: true, separatorChar: ',');

            //Build data
            var pipeline = context.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var pipelineOnlineGradient = context.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(context.Regression.Trainers.OnlineGradientDescent());

            var pipelineSdca = context.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(context.Regression.Trainers.OnlineGradientDescent());

            var modelPoisonRegression = pipeline.Fit(trainData);

            var modelGradient = pipelineOnlineGradient.Fit(trainData);

            var modelSdca = pipelineSdca.Fit(trainData);

            //Evaluate 
            var predictionsPoison = modelPoisonRegression.Transform(trainData);
            var matricesPoison = context.Regression.Evaluate(predictionsPoison);

            var predictionsGradient = modelGradient.Transform(trainData);
            var matricesGradient = context.Regression.Evaluate(predictionsGradient);

            var predictionsSdca = modelSdca.Transform(trainData);
            var matricesSdca = context.Regression.Evaluate(predictionsSdca);

            Console.WriteLine($"PR RSquared: {matricesPoison.RSquared}, Gradient RSquared: {matricesGradient.RSquared}, Sdca RSquared: {matricesSdca.RSquared}");
            //Predict
            var newData = new SalaryData
            {
                YearsOfExperience = 1.2f
            };
            var predictionFuncPoison = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(modelPoisonRegression);
            var predictionPoison = predictionFuncPoison.Predict(newData);

            var predictionFuncGradient = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(modelGradient);
            var predictionGradient = predictionFuncPoison.Predict(newData);

            var predictionFuncSdca = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(modelSdca);
            var predictionSdca = predictionFuncSdca.Predict(newData);

            Console.WriteLine($"Poison Pred: {predictionPoison.PredictedSalary}, Gredient Pred: {predictionGradient.PredictedSalary}, Sdca Prediction: {predictionSdca.PredictedSalary}");

            Console.ReadLine();
        }
    }
}
