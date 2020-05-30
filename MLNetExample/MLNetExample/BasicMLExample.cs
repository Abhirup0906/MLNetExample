using Microsoft.ML;
using MLNetExample.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetExample
{
    public static class BasicMLExample
    {
        public static void ExecuteBasicML()
        {
            var context = new MLContext();

            //Load data
            var trainData = context.Data.LoadFromTextFile<SalaryData>("Data//SalaryData.csv", hasHeader: true, separatorChar: ',');

            //Build data
            var pipeline = context.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var modelPoisonRegression = pipeline.Fit(trainData);

            //Evaluate 
            var predictionsPoison = modelPoisonRegression.Transform(trainData);
            var matricesPoison = context.Regression.Evaluate(predictionsPoison);

            Console.WriteLine($"PR RSquared: {matricesPoison.RSquared}");
            //Predict
            var newData = new SalaryData
            {
                YearsOfExperience = 1.2f
            };
            var predictionFuncPoison = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(modelPoisonRegression);
            var predictionPoison = predictionFuncPoison.Predict(newData);

            Console.WriteLine($"Poison Pred: {predictionPoison.PredictedSalary}");
        }
    }
}
