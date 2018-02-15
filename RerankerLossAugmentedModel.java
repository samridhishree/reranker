package edu.berkeley.nlp.assignments.rerank.student;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import edu.berkeley.nlp.assignments.rerank.LossAugmentedLinearModel;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.IntCounter;

public class RerankerLossAugmentedModel implements LossAugmentedLinearModel<FeatureData>
{
	// returns everything an SVM trainer needs to do its thing for a given
	  // training datum
	  // datum: current training datum... including gold label... you get to define
	  // the type T
	  // goldFeatures: feature vector of correct label for current training datum
	  // lossAugGuessFeatures: feature vector of loss-augmented guess using weights
	  // provided for current training datum
	  // lossOfGuess: loss of loss-augmented guess compared to gold label for
	  // current training datum
	  public UpdateBundle getLossAugmentedUpdateBundle(FeatureData datum, IntCounter weights)
	  {
		  IntCounter goldFeatures = datum.getGoldFeatures();
		  double maxVal = Double.NEGATIVE_INFINITY;
		  IntCounter argmax = null;
		  double lossOfGuess, curVal;
		  lossOfGuess = curVal =  0;
		  double curLoss = 0;
		  double[] kBestLoss = datum.getkBestLoss();
		  //IntCounter[] kBestFeatures = datum.getkBestFeatures();
		  int[][] kBestFeatures = datum.getkBestFeatures();
		   
		  //Get the loss augmented feature vector and its corresponding loss.
		  for(int idx = 0; idx < kBestLoss.length; idx++)
		  {
			  //Tree<String> curTree = datum.getkParseTrees().get(idx);
			  IntCounter curFeature = CreateFeatureCounterList(kBestFeatures[idx]);
			  curVal = curFeature.dotProduct(weights);
			  //curLoss = (kBestLoss[idx])?1:0;
			  curLoss = kBestLoss[idx];
			  
			  if((curVal+curLoss) > maxVal)
			  {
				  maxVal = curVal+curLoss;
				  argmax = curFeature;
				  lossOfGuess = curLoss;
			  }
		  }
		  
		  UpdateBundle bundle = new UpdateBundle(goldFeatures, argmax, lossOfGuess);
		  return bundle;
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
	  
	  /*
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
	  */
}
