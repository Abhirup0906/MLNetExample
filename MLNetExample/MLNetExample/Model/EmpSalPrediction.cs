using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetExample.Model
{
    public class EmpSalPrediction
    {
        [ColumnName("Score")]
        public float Gross { get; set; }
    }
}
