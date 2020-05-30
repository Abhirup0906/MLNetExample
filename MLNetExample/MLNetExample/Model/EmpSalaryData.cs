using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetExample.Model
{
    public class EmpSalaryData
    {
        [LoadColumn(0)]
        public string Name { get; set; }
        [LoadColumn(1)]
        public string JobTitle { get; set; }
        [LoadColumn(2)]
        public string DeptId { get; set; }
        [LoadColumn(3)]
        public string Descr { get; set; }
        [LoadColumn(4)]
        public DateTime Hire_Dt { get; set; }
        [LoadColumn(5)]
        public float Annual_Rt { get; set; }
        [LoadColumn(6)]
        public float Gross { get; set; }
    }
}
