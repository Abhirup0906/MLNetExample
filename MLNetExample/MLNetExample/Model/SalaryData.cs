using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetExample.Model
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsOfExperience { get; set; }

        [LoadColumn(1), ColumnName("Label")]
        public float Salary { get; set; }
    }
}
