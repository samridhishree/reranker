package edu.berkeley.nlp.assignments.rerank.student;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.SurfaceHeadFinder;
import edu.berkeley.nlp.ling.AnchoredTree;
import edu.berkeley.nlp.ling.Constituent;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;

/**
 * Baseline feature extractor for k-best lists of parses. Note that this does
 * not implement Featurizer, though you can adapt it to do so.
 *
 */
public class ComplexFeatureExtractor 
{
	Set<String> punctutationList = new HashSet<String>(Arrays.asList(new String[] {"'", ".", "\"", "?", "!", "}", "]", ")", ":", ";" , ","}));
/**
   * 
   * @param kbestList
   * @param idx
   *          The index of the tree in the k-best list to extract features for
   * @param featureIndexer
   * @param addFeaturesToIndexer
   *          True if we should add new features to the indexer, false
   *          otherwise. When training, you want to make sure you include all
   *          possible features, but adding features at test time is pointless
   *          (since you won't have learned weights for those features anyway).
   * @return the list of features present in the tree indexed by parameter idx.
   */
  public int[] ExtractKListFeatures(KbestList kbestList, int idx, Indexer<String> featureIndexer, HashMap<Integer, Integer> featureCounter, boolean addFeaturesToIndexer) 
  {
	  Tree<String> tree = kbestList.getKbestTrees().get(idx);
	  return ExtractTreeFeatures(tree, idx,featureIndexer, featureCounter, addFeaturesToIndexer);  
  }
  
  public int[] ExtractGoldTreeFeatures(Tree<String> goldTree, KbestList kbestList, Indexer<String> featureIndexer, HashMap<Integer, Integer> featureCounter) 
  {
	  int idx = -1;
	  for(int i = 0; i < kbestList.getKbestTrees().size(); i++)
	  {
		  Tree<String> curTree = kbestList.getKbestTrees().get(i);
		  if(curTree.toString().equals(goldTree.toString()))
		  {
			  idx = i;
			  break;
		  }
	  }
      return ExtractTreeFeatures(goldTree, idx, featureIndexer, featureCounter, true);
  }
  
  public int[] ExtractTreeFeatures(Tree<String> tree, int idx, Indexer<String> featureIndexer, HashMap<Integer, Integer> featureCounter, boolean addFeaturesToIndexer)
  {
	// Converts the tree
    // (see below)
    AnchoredTree<String> anchoredTree = AnchoredTree.fromTree(tree);
    // If you just want to iterate over labeled spans, use the constituent list
    Collection<Constituent<String>> constituents = tree.toConstituentList();
    // You can fire features on parts of speech or words
    List<String> poss = tree.getPreTerminalYield();
    List<String> words = tree.getYield();
    // Allows you to find heads of spans of preterminals. Use this to fire
    // dependency-based features
    // like those discussed in Charniak and Johnson
    SurfaceHeadFinder shf = new SurfaceHeadFinder();
    

    // FEATURE COMPUTATION
    List<Integer> feats = new ArrayList<Integer>();
    // Fires a feature based on the position in the k-best list. This should
    // allow the model to learn that
    // high-up trees
    if(idx != -1)
    	addFeature("Posn=" + idx, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
    //if(score != Double.POSITIVE_INFINITY)
    //{
    	//int scoreNum = GetScoreBucketNumber(score);
    	//addFeature("Score=" + scoreNum, feats, featureIndexer, addFeaturesToIndexer);
   // }
    	

    for (AnchoredTree<String> subtree : anchoredTree.toSubTreeList()) 
    {
      if (!subtree.isPreTerminal() && !subtree.isLeaf()) 
      {
    	String rule = "Rule=" + subtree.getLabel() + " ->";
    	String spiltRule = subtree.getLabel() + "->";
    	List<AnchoredTree<String>> children = subtree.getChildren();
        int startIndex = subtree.getStartIdx();
        int lastIndex = subtree.getEndIdx() - 1; 
        String headLabel = subtree.getLabel();
        
        if(children.size() == 2)
        {
        	AnchoredTree<String> firstChild = children.get(0);
        	AnchoredTree<String> secondChild = children.get(1);
        	String splitWord = words.get(firstChild.getEndIdx()-1);
        	spiltRule = spiltRule + "(" + firstChild.getLabel() + ".." + splitWord + ")" + secondChild.getLabel() + ")";
        	addFeature(spiltRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        }
        
        //Span shape 
        String spanShape = subtree.getLabel() + "->";
        for(int i=startIndex; i<=lastIndex; i++)
        { 
        	String curWord = words.get(i);
        	char firstChar = curWord.charAt(0);
        	
        	if(Character.isUpperCase(firstChar))
        		spanShape += "X";
        	else if(Character.isLowerCase(firstChar))
        		spanShape += "x";
        	else if(Character.isDigit(firstChar))
        		spanShape += "0";
        	else
        		spanShape += firstChar;
        		
        
        }
        addFeature(spanShape, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        
        // Fires a feature based on the identity of a nonterminal rule
        // production. This allows the model to learn features
        // roughly equivalent to those in an unbinarized coarse grammar.
        int numChild = 0;
        for (AnchoredTree<String> child : children) 
        {
        	rule += " " + child.getLabel();
        	numChild++;
        }
        addFeature(rule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        //String numChildren = headLabel + "numChildren=" + numChild;
        //addFeature(numChildren, feats, featureIndexer, addFeaturesToIndexer);
        
        if((startIndex-1) >= 0)
        {
        	String firstWordContextRule = rule + "^firstcontext= " + words.get(startIndex-1);
        	addFeature(firstWordContextRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        }
        if((lastIndex+1) < words.size())
        {
        	String lastWordContextRule = rule + "^lastcontext= " + words.get(lastIndex+1);
        	addFeature(lastWordContextRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        }

        /*
         * Add features for span lengths for non-terminal heads.
         * The lengths are bucketized (1,2,3,4,5,10,20,21)
         */
        String spanLengthRule = GetTheSpanLengthRule(rule, subtree);
        addFeature(spanLengthRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        
        /*
         * Add the w-edges features. Adds the left span word and
         * right span word to each non terminal label. Also creates similar
         * features for the pos tags of those words.
         */
        String firstWordRule = rule + "^first=" + words.get(startIndex);
        String lastWordRule = rule + "^last=" + words.get(lastIndex);
        String firstPosRule = rule + "^posfirst=" + poss.get(startIndex);
        String lastPosRule = rule + "^poslast=" + poss.get(lastIndex);
        addFeature(firstWordRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        addFeature(lastWordRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        addFeature(firstPosRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        addFeature(lastPosRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer); 
        
        int index = GetNonPunctuationIndex(words);
        //if(index != -1)
        //{
        	int length = GetRightBranchLength(poss.get(index), 0, anchoredTree, index);
        	String rightBranchRule = "rb_length=" + length;
        	addFeature(rightBranchRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
        //}
       
        
        for(int i=0; i<(children.size()-2); i++)
  	  	{
        	String ngramRule = headLabel + "->";
        	for(int j=i; j<i+3; j++)
        	{
        		ngramRule =  ngramRule + "_" + children.get(j).getLabel();
        	}
        	addFeature(ngramRule, feats, featureIndexer, featureCounter, addFeaturesToIndexer);
  	  	}
        
        //Surface head annotations
        /*
        List<String> preterminals = new ArrayList<String>();
        for(int i = startIndex; i<=lastIndex; i++)
        	preterminals.add(poss.get(i));
        int synctacticHead = shf.findHead(headLabel, preterminals);
        String headRule = headLabel +"^" + "synctacticHead=" + poss.get(synctacticHead);
        String lexHeadRule = headLabel +"^" + "lexicalHead=" + words.get(synctacticHead);
        addFeature(headRule, feats, featureIndexer, addFeaturesToIndexer);
        addFeature(lexHeadRule, feats, featureIndexer, addFeaturesToIndexer);
        */
        
      }
    }
    
    int[] featsArr = new int[feats.size()];
    for (int i = 0; i < feats.size(); i++) {
      featsArr[i] = feats.get(i).intValue();
    }
    return featsArr;
  }

  /**
   * Shortcut method for indexing a feature and adding it to the list of
   * features.
   * 
   * @param feat
   * @param feats
   * @param featureIndexer
   * @param addNew
   */
  private void addFeature(String feat, List<Integer> feats, Indexer<String> featureIndexer, HashMap<Integer, Integer> featureCounter, boolean addNew) 
  {
    if (addNew || featureIndexer.contains(feat)) 
    {
    	int featureIndex = featureIndexer.addAndGetIndex(feat);
	    feats.add(featureIndex);
	    if(addNew)
	    {
	    	int val = 1;
	    	if(featureCounter.containsKey(featureIndex))
	    		val = featureCounter.get(featureIndex) + 1;
	    	featureCounter.put(featureIndex, val);
	   }
    }
  }
  
  private String GetTheSpanLengthRule(String rule, AnchoredTree<String> subTree)
  {
	  int spanLength = subTree.getSpanLength();
	  int bucket = GetSpanLengthBucketNumber(spanLength);
	  String lenRule = rule + "^spanlength=" + bucket;
	  return lenRule;
  }
  
  private int GetSpanLengthBucketNumber(int length)
  {
	  int num = 0;
	  if((length >=1) && (length <= 5))
		  num = length;
	  else if(length <= 10)
		  num = 10;
	  else if(length <= 20)
		  num = 20;
	  else
		  num = 21;
	  return num;
  }
  
  private int GetRightBranchBucketNumber(int length)
  {
	  int num = 0;
	  if(length <= 10)
		  num = 10;
	  else if(length <= 20)
		  num = 20;
	  else
		  num = 30;
	  return num;
  }
  
  private int GetScoreBucketNumber(double score)
  {
	  int num;
	  if(score >= -9)
		  num = 9;
	  else if(score >= -11.5)
		  num = 11;
	  else if(score >= -13)
		  num = 13;
	  else if(score >= -14.4)
		  num = 14;
	  else if(score >= -15.5)
		  num = 15;
	  else if(score >= -16.8)
		  num = 16;
	  else if(score >= -18.2)
		  num = 18;
	  else if(score >= -20.1)
		  num = 20;
	  else if(score >= -23.5)
		  num = 23;
	  else
		  num = 120;
	  return num;
  }
  
  private int GetRightBranchLength(String preTerminal, int len, AnchoredTree<String> tree, int index)
  {
	  if(tree.getStartIdx() == (tree.getEndIdx()-1) &&
		(tree.getLabel().equals(preTerminal)))
	  {
		  return (len+1);
	  }
	  
	  List<AnchoredTree<String>> children = tree.getChildren();
	  for(int i = (children.size()-1); i >= 0; i--)
	  {
		  if((children.get(i).getEndIdx() > index) &&
			 (children.get(i).getStartIdx() <= index))
		  {
			  return GetRightBranchLength(preTerminal, len+1, children.get(i), index);
		  }
	  }
	  return 0;
  }
  
  private int GetNonPunctuationIndex(List<String> words)
  {
	  for(int i=(words.size()-1); i>=0; i--)
	  {
		  if(!punctutationList.contains(words.get(i)))
			  return i;
	  }
	  return 0;
  }
}
