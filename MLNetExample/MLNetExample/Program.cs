using System;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using MLNetExample.Model;

namespace MLNetExample
{
    class Program
    {
        static void Main(string[] args)
        {
            //OneHotEncoding.Example();
            EmpSalMLExample.ExecuteEmpSalaryML();
            Console.ReadLine();
        }
    }
}
