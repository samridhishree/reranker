package edu.berkeley.nlp.assignments.rerank.student;

import java.util.HashMap;
import java.util.List;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.IntCounter;

/*
 * This class provides the basic structure for datum.
 * SVM learner and the LossAugmentedLinearModel uses this.
 */

public class FeatureData 
{
	private IntCounter goldFeatures;
	//private boolean kBestLoss[];
	private double kBestLoss[];
	private int[][] kFeatures;
	//private IntCounter kBestFeatures[];
	//private Tree<String> goldTree;
	//private List<Tree<String>> kParseTrees;
	//private List<IntCounter> kBestFeatures;	
	
	public FeatureData(IntCounter goldFeatures, double[] kBestLoss, int[][] kBestFeatures)
	{
		this.setGoldFeatures(goldFeatures);
		this.setkBestLoss(kBestLoss);
		this.setkBestFeatures(kBestFeatures);
		//this.setkBestFeatures(kBestFeatures);
		//this.setkParseTreeFeatures(kParseTreeFeatures);
		//this.setGoldTree(goldTree);
		//this.setkPar	seTrees(kParseTrees);
	}

	public int[][] getkBestFeatures() {
		return kFeatures;
	}

	public void setkBestFeatures(int[][] kBestFeatures) {
		this.kFeatures = kBestFeatures;
	}
	
	public IntCounter getGoldFeatures() {
		return goldFeatures;
	}

	public void setGoldFeatures(IntCounter goldFeatures) {
		this.goldFeatures = goldFeatures;
	}

	public double[] getkBestLoss() {
		return kBestLoss;
	}

	public void setkBestLoss(double kBestLoss[]) {
		this.kBestLoss = kBestLoss;
	}
	
	/*
	public List<Tree<String>> getkParseTrees() {
		return kParseTrees;
	}

	public void setkParseTrees(List<Tree<String>> kParseTrees) {
		this.kParseTrees = kParseTrees;
	}
	public Tree<String> getGoldTree() {
		return goldTree;
	}

	public void setGoldTree(Tree<String> goldTree) {
		this.goldTree = goldTree;
	}
	
	public IntCounter[] getkBestFeatures() {
		return kBestFeatures;
	}

	public void setkBestFeatures(IntCounter[] kBestFeatures) {
		this.kBestFeatures = kBestFeatures;
	}
	*/
}
