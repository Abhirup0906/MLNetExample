using Microsoft.ML;
using Microsoft.ML.Data;
using MLNetExample.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Transforms;

namespace MLNetExample
{
    public static class EmpSalMLExample
    {
        public static void ExecuteEmpSalaryML()
        {
            //create ML with seed for repeatable/deterministic results
            var context = new MLContext(seed:0);

            //Load Data
            var trainingData = context.Data.LoadFromTextFile<EmpSalaryData>("Data//baltimore-city-employee-salaries-fy2019-1.csv", hasHeader: true, separatorChar: ',');

            //replace Nan values
            var replaceNanAnnual = context.Transforms.DropColumns(nameof(EmpSalaryData.Annual_Rt));
            var transformAnnual = replaceNanAnnual.Fit(trainingData);
            var tranformedAnnualData = transformAnnual.Transform(trainingData);

            var replaceNanGross = context.Transforms.ReplaceMissingValues(nameof(EmpSalaryData.Gross), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Minimum);
            var transformNanGross = replaceNanGross.Fit(tranformedAnnualData);
            var transformedGrossData = transformNanGross.Transform(tranformedAnnualData);
                       
            var maxGross = transformedGrossData.GetColumn<float>(nameof(EmpSalaryData.Gross)).Max();
            var minGross = transformedGrossData.GetColumn<float>(nameof(EmpSalaryData.Gross)).Min();
                        
            var removedGrossOutLier = context.Data.FilterRowsByColumn(transformedGrossData, nameof(EmpSalaryData.Gross), lowerBound: minGross, upperBound: maxGross);

            var nameDelEsti = context.Transforms.DropColumns(nameof(EmpSalaryData.Name));
            var transformDelName = nameDelEsti.Fit(removedGrossOutLier);
            var transDelNameData = transformDelName.Transform(removedGrossOutLier);

            var descDelEsti = context.Transforms.DropColumns(nameof(EmpSalaryData.Descr));
            var transDescDel = descDelEsti.Fit(transDelNameData);
            var transDescDelData = transDescDel.Transform(transDelNameData);

            var tainTest = context.Data.TrainTestSplit(transDescDelData, 0.1);
            
            var dataPipeline = context.Transforms.CopyColumns(outputColumnName: "Label", nameof(EmpSalaryData.Gross))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "JobTitleEncoded", inputColumnName: nameof(EmpSalaryData.JobTitle)))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "DeptIdEncoded", inputColumnName: nameof(EmpSalaryData.DeptId)))
                .Append(context.Transforms.Concatenate("Features", "JobTitleEncoded", "DeptIdEncoded"));

            var trainer = context.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataPipeline.Append(trainer);

            var trainingModel = trainingPipeline.Fit(tainTest.TrainSet);

            IDataView predictions = trainingModel.Transform(tainTest.TestSet);
            var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            
            Console.WriteLine($"PR RSquared: {metrics.RSquared}");
        }
    }
}
