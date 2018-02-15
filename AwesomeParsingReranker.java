package edu.berkeley.nlp.assignments.rerank.student;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.LossAugmentedLinearModel;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.PrimalSubgradientSVMLearner;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

/**
 * k-best discriminative reranker
 * 
 * @author Samridhi
 * Uses SVM to learn the feature weights.
 *
 */

public class AwesomeParsingReranker implements ParsingReranker 
{
	IntCounter weightVector;
	Indexer<String> featureIndexer;
	List<FeatureData> lossAugmentedTrainData;
	HashMap<Integer, Integer> featureCounter;
	int threshold = 5;
	Indexer<String> prunedFeatureIndexer;
	
	/*
	 * Class constructor for training.
	 */
	public AwesomeParsingReranker(Iterable<Pair<KbestList,Tree<String>>> kbestListsAndGoldTrees)
	{
		weightVector = new IntCounter();
		featureIndexer = new Indexer<String>();
		lossAugmentedTrainData = new ArrayList<FeatureData>();
		featureCounter = new HashMap<Integer, Integer>();
		prunedFeatureIndexer = new Indexer<String>();
		
		
		/*
		 * Extract the features for each tree in the k-best list and build the cache
		 */
		FeatureExtractor featExtract = new FeatureExtractor(); 
		ComplexFeatureExtractor complexFeatExtract = new ComplexFeatureExtractor();
		List<FeatureData> trainingDataList = new ArrayList<FeatureData>();
		
		//First Pass
		/*
		for(Pair<KbestList,Tree<String>> iter : kbestListsAndGoldTrees)
		{
			KbestList kbestList = iter.getFirst();
			List<Tree<String>> kBestTrees = kbestList.getKbestTrees();
			int kListSize = kBestTrees.size();
			Tree<String> goldTree = iter.getSecond();
			
			// Get the feature vector for goldtree
			complexFeatExtract.ExtractGoldTreeFeatures(goldTree, kbestList, featureIndexer, featureCounter);
			
			// Get the feature vectors corresponding to each parse tree
			for(int i = 0; i < kBestTrees.size(); i++)
			{
				complexFeatExtract.ExtractKListFeatures(kbestList, i, featureIndexer, featureCounter, true);
			}
		}
		*/
		
		//UpdatePrunedFeatureIndexer();
		
		//Second Pass
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
				kBestLoss[i] = GetLossFunctionValue(goldTree, kBestTrees.get(i), true);
			}
			FeatureData trainData = new FeatureData(goldFeatures, kBestLoss, featureList);
			trainingDataList.add(trainData);
		}
		
		
		
		//Learn the weight vector with primal SVM
		double stepSize  = 1e-3;
		double regConstant = 1e-4;
		int batchSize = 100;
		int numIters = 40;
		
		System.out.println("Feature indexer size = " + featureIndexer.size());
		PrimalSubgradientSVMLearner<FeatureData> svmLearner = new PrimalSubgradientSVMLearner<FeatureData>(stepSize, 
																	regConstant, featureIndexer.size(), batchSize);
		
		RerankerLossAugmentedModel lossModel = new RerankerLossAugmentedModel();
		weightVector = svmLearner.train(weightVector, lossModel, trainingDataList, numIters);	
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
	
	public List<FeatureData> PruneFeatures(List<FeatureData> trainingDataList)
	{
		List<FeatureData> newTrainDataList = new ArrayList<FeatureData>();
		for(FeatureData data : trainingDataList)
		{
			int[][] newFeatureList = new int[data.getkBestFeatures().length][];
			for(int i = 0; i < data.getkBestFeatures().length; i++)
			{
				int[] curFeature = data.getkBestFeatures()[i];
				List<Integer> newList = new ArrayList<Integer>();
				for(int j = 0; j< curFeature.length; j++)
				{
					if(featureCounter.get(curFeature[j]) >= threshold)
					{
						newList.add(curFeature[j]);
						prunedFeatureIndexer.add(featureIndexer.get(curFeature[j]));
					}
				}
				newFeatureList[i] = new int[newList.size()];
				//newFeatureList[i] = newList.stream().mapToInt(k->k).toArray();
				for(int k = 0; k<newList.size(); k++)
					newFeatureList[i][k] = newList.get(k);
			}
			
			//Update gold features
			IntCounter curGoldFeature = data.getGoldFeatures();
			IntCounter newGoldList = new IntCounter();
			
			for(int curFeat : curGoldFeature.keySet())
			{
				if(featureCounter.get(curFeat) >= threshold)
				{
					newGoldList.put(curFeat, curGoldFeature.get(curFeat));
					prunedFeatureIndexer.add(featureIndexer.get(curFeat));
				}
			}
			
			FeatureData newTrainData = new FeatureData(newGoldList, data.getkBestLoss(), newFeatureList);
			newTrainDataList.add(newTrainData);
		}
		return newTrainDataList;
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
	
	private void UpdatePrunedFeatureIndexer()
	{
		for(int feat = 0; feat < featureIndexer.size(); feat++)
		{
			if(featureCounter.get(feat) >= 5)
				prunedFeatureIndexer.add(featureIndexer.get(feat));
		}
	}
}
