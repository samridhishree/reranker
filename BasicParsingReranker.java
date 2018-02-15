package edu.berkeley.nlp.assignments.rerank.student;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

/**
 * k-best discriminative reranker
 * 
 * @author Samridhi
 * Uses Perceptron to learn the feature weights.
 */

public class BasicParsingReranker implements ParsingReranker 
{
	IntCounter weightVector;
	Indexer<String> featureIndexer;
	List<FeatureData> lossAugmentedTrainData;
	
	/*
	 * Class constructor for training.
	 */
	public BasicParsingReranker(Iterable<Pair<KbestList,Tree<String>>> kbestListsAndGoldTrees)
	{
		featureIndexer = new Indexer<String>();
		lossAugmentedTrainData = new ArrayList<FeatureData>();
		
		/*
		 * Extract the features for each tree in the k-best list and build the cache
		 */
		FeatureExtractor featExtract = new FeatureExtractor(); 
		List<FeatureData> trainingDataList = new ArrayList<FeatureData>();
		System.out.println("Extracting Features");
		
		for(Pair<KbestList,Tree<String>> iter : kbestListsAndGoldTrees)
		{
			KbestList kbestList = iter.getFirst();
			List<Tree<String>> kBestTrees = kbestList.getKbestTrees();
			int kListSize = kBestTrees.size();
			Tree<String> goldTree = iter.getSecond();
			double[] kBestLoss = new double[kListSize];
			
			// Get the feature vector for goldtree
			int[] goldList = featExtract.ExtractGoldTreeFeatures(goldTree, kbestList, featureIndexer, true);
			IntCounter goldFeatures = CreateFeatureCounterList(goldList);
			int[][] featureList = new int[kBestTrees.size()][];
			
			// Get the feature vectors corresponding to each parse tree
			for(int i = 0; i < kBestTrees.size(); i++)
			{
				int[] curFeatureList = featExtract.ExtractKListFeatures(kbestList, i, featureIndexer, true);
				featureList[i] = curFeatureList;
				kBestLoss[i] = GetLossFunctionValue(goldTree, kBestTrees.get(i), false);
			}
			FeatureData trainData = new FeatureData(goldFeatures, kBestLoss, featureList);
			trainingDataList.add(trainData);
		}
		
		IntCounter tempWeightVector = new IntCounter(featureIndexer.size());
		for(int i = 0; i < featureIndexer.size(); i++)
		{
			tempWeightVector.put(i, 0);
		}
		
		System.out.println("Training with perceptron");
		int maxIter = 30;
		double tolerance = 0.05;
		System.out.println("Norm squared before calling the perceptron = " + tempWeightVector.normSquared());
		PerceptronLearner trainer = new PerceptronLearner(maxIter, tolerance);
		weightVector = trainer.Train(tempWeightVector, trainingDataList);
	}
	
	public IntCounter CreateFeatureCounterList(int[] featureList)
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
	
	public Tree<String> getBestParse(List<String> sentence, KbestList kbestList) 
	{
		double maxVal, curVal;
		Tree<String> argmax;
		maxVal = Double.NEGATIVE_INFINITY;
		curVal = 0;
		argmax = null;
		FeatureExtractor featExtract = new FeatureExtractor(); 
		List<Tree<String>> kBestTrees = kbestList.getKbestTrees();
		
		for(int idx = 0; idx < kBestTrees.size(); idx++)
		{
			int[] curFeature = featExtract.ExtractKListFeatures(kbestList, idx, featureIndexer, false);
			IntCounter featVector = CreateFeatureCounterList(curFeature);
			curVal = featVector.dotProduct(weightVector);
			if(curVal > maxVal)
			{
				maxVal = curVal;
				argmax = kBestTrees.get(idx);
			}
		}
		return argmax;
	}
	
  private double GetLossFunctionValue(Tree<String> goldTree, Tree<String> guessTree, Boolean useF1)
  {
	  double lossValue = 0;
	  if(useF1)
	  {
		  EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = new EnglishPennTreebankParseEvaluator
					.LabeledConstituentEval<String>(Collections.singleton("ROOT"),
						new HashSet<String>(Arrays.asList(new String[] { "''", "``", ".", ":", "," })));
		  double f1 = eval.evaluateF1(guessTree, goldTree);
		  lossValue = 1-f1;
	  }
	  else
	  {
		  lossValue = ((guessTree.toString()).equalsIgnoreCase(goldTree.toString()))? 0 : 1;
	  }
	  return lossValue;
  }
}
