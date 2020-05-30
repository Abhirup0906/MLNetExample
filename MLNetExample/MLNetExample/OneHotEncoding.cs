using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetExample
{
    public static class OneHotEncoding
    {
        public static void Example()
        {
            var mlContext = new MLContext();

            //create dataset
            var samples = new[] {
                new DataPoint { Name="test1", Education = "0-5yrs", ZipCode = "98005" },
                new DataPoint {Name="test2", Education = "0-5yrs", ZipCode = "98052" },
                new DataPoint { Name="test3", Education = "6-11yrs", ZipCode = "98005" },
                new DataPoint {Name="test4", Education = "6-11yrs", ZipCode = "98052"},
                new DataPoint { Name="test5", Education = "11-15yrs", ZipCode = "98005"}
            };

            //Convert training data into data view
            IDataView dataView = mlContext.Data.LoadFromEnumerable(samples);

            // Multi column example: A pipeline for one hot encoding two columns
            // 'Education' and 'ZipCode'.
            var multiColumnKeyPipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] {
                new InputOutputColumnPair("Education"), new InputOutputColumnPair("ZipCode")
            });

            //fit and transform
            IDataView transformedData = multiColumnKeyPipeline.Fit(dataView).Transform(dataView);
            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);

            Console.WriteLine(
                "One Hot Encoding of two columns 'Education' and 'ZipCode'.");

            // One Hot Encoding of two columns 'Education' and 'ZipCode'.

            foreach (TransformedData item in convertedData)
                Console.WriteLine("{0}\t\t\t{1}\t\t\t{2}", item.Name, string.Join(" ", item.Education),
                    string.Join(" ", item.ZipCode));
        }

        private class DataPoint
        {
            public string Name { get; set; }
            public string Education { get; set; }

            public string ZipCode { get; set; }
        }

        private class TransformedData
        {
            public string Name { get; set; }
            public float[] Education { get; set; }

            public float[] ZipCode { get; set; }
        }
    }


}
