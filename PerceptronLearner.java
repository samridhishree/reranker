package edu.berkeley.nlp.assignments.rerank.student;

import java.util.List;
import edu.berkeley.nlp.util.IntCounter;

public class PerceptronLearner
{
	private int numOfIterations;
	private double tolerance;
	double lamda = 0.5;
	
	public PerceptronLearner(int maxIter, double toleranceVal)
	{
		numOfIterations = maxIter;
		tolerance = toleranceVal;
	}
	
	public IntCounter Train(IntCounter initWeight, List<FeatureData> datum)
	{
		System.out.println("In Train with num iterations = " + numOfIterations);
		System.out.println("Data size = " + datum.size());
		IntCounter avgWeightVector = new IntCounter();
		IntCounter curWeightVector = initWeight;
		avgWeightVector = CopyAndAverageVectors(avgWeightVector, initWeight);
		
		for(int i = 1; i <= numOfIterations; i++)
		{
			System.out.println("Iteration Num = " + i);
			
			int num = 1;
			for(FeatureData data : datum)
			{
				IntCounter goldFeatures = data.getGoldFeatures();
				double[] kBestLoss = data.getkBestLoss();
				int[][] kBestFeatures = data.getkBestFeatures();
				
				int argmaxIndex = GetArgMaxTreeIndex(kBestFeatures, curWeightVector);
				
				if(kBestLoss[argmaxIndex] != 0)
				{
					IntCounter argmaxFeature = CreateFeatureCounterList(kBestFeatures[argmaxIndex]);
					IntCounter mistakeVector = GetMistakeVector(goldFeatures, argmaxFeature);
					curWeightVector = UpdateWeight(curWeightVector, mistakeVector);
					if(num == 10000)
					{
						avgWeightVector = CopyAndAverageVectors(avgWeightVector, curWeightVector);
						num = 1;
					}
				}
				num++;
			}
			
			System.out.println("Current weight vector norm squared = " + curWeightVector.normSquared());
			System.out.println("Current average weight vector norm squared out of the function = " + avgWeightVector.normSquared());
		}
		//return curWeightVector;
		return avgWeightVector;
	}
	
	private IntCounter CopyAndAverageVectors(IntCounter avgVector, IntCounter curVector)
	{
		IntCounter weightVector = avgVector;
		double val;
		for(int key : curVector.keySet())
		{
			val = (lamda * weightVector.get(key)) + ((1-lamda) * (curVector.get(key)));
			weightVector.put(key, val);
		}
		return weightVector;
	}
	
	/*
	private Boolean CheckConvergence(IntCounter oldWeight, IntCounter newWeight)
	{
		//System.out.println("In check convergence");
		double value = oldWeight.normSquared();
		double nextValue = newWeight.normSquared();
		System.out.println("value = " + value);
		System.out.println("nextVal = " + nextValue);
		double EPS = 1e-10;
		if (value == nextValue) 
			return true;
		double valueChange = SloppyMath.abs(nextValue - value);
		double valueAverage = SloppyMath.abs(nextValue + value + EPS) / 2.0;
		//System.out.println("The valuechange/valueaverage = " + (valueChange / valueAverage));
		if (valueChange / valueAverage < tolerance) 
			return true;
		
		return false;
	}
	*/
	
	private IntCounter GetMistakeVector(IntCounter goldFeat, IntCounter treeFeat)
	{
		IntCounter mistakeVector = new IntCounter();
		
		for(int key : goldFeat.keySet())
		{
			mistakeVector.put(key, goldFeat.get(key));
		}
		
		for(int key : treeFeat.keySet())
		{
			double value = (goldFeat.get(key) - treeFeat.get(key));
			mistakeVector.put(key, value);
		}
		return mistakeVector;
	}
	
	private IntCounter UpdateWeight(IntCounter initWeight, IntCounter mistakeVector)
	{
		IntCounter weightVector = initWeight;
		
		for(int key : mistakeVector.keySet())
		{
			double val = weightVector.get(key) + mistakeVector.get(key);
			weightVector.put(key, val);
		}
		return weightVector;
	}
	
	private int GetArgMaxTreeIndex(int[][] kBestFeatures, IntCounter curWeight)
	{
		double curVal, maxVal;
		curVal = 0;
		maxVal = Double.NEGATIVE_INFINITY;
		int argmaxIndex = -1;
		IntCounter curFeature = new IntCounter();
		
		for(int idx = 0; idx < kBestFeatures.length; idx++)
		{
			curFeature = CreateFeatureCounterList(kBestFeatures[idx]);
			curVal = curFeature.dotProduct(curWeight);
			if(curVal > maxVal)
			{
				maxVal = curVal;
				argmaxIndex = idx;
			}
		}
		return argmaxIndex;
	}
	
	private IntCounter CreateFeatureCounterList(int[] featureList)
	{
		IntCounter finalList = new IntCounter();
		
		//Update the counts for the occurrences of a feature
		for(int curFeature : featureList)
		{
			double val = finalList.get(curFeature) + 1;
			finalList.put(curFeature, val);
		}
		return finalList;
	}
}
