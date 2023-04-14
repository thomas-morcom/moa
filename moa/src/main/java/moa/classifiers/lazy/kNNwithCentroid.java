/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.lazy;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;

import java.util.Arrays;

/**
 * k Nearest Neighbor.<p>
 *
 * Valid options are:<p>
 *
 * -k number of neighbours <br> -m max instances <br> 
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version 03.2012
 */
public class kNNwithCentroid extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 1, 1, 1);

	// For checking regression with mean value or median value
	public FlagOption medianOption = new FlagOption("median",'m',"median or mean");
	public FlagOption debugFlag = new FlagOption("debug",'d',"Debug");

	public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

        public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
                "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                "KDTree search algorithm for nearest neighbour search"
            }, 0);


	int C = 0;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window;
	protected Instances centroids;
	protected double[][] attrSums;
	protected int[] classCounters;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
			this.centroids = new Instances(context,0);
			this.centroids.setClassIndex(context.classIndex());
			this.attrSums = new double[2][context.numAttributes()];
			this.classCounters = new int[2];
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void resetLearningImpl() {
		this.window = null;
		this.centroids = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		if (inst.classValue() > C)
			C = (int)inst.classValue();

		// Creating the different arrays and instances, if they aren't created already
		if (this.window == null) {
			this.window = new Instances(inst.dataset());
			this.centroids = new Instances(inst.dataset());
			this.attrSums = new double[2][inst.numAttributes()];
			this.classCounters = new int[2];
		}

		// Adding the very first instance to both classes
		// they are a nothing instance so that they do not have any sway on the classification
		if (this.centroids.size() == 0){
			DenseInstance initialInst = new DenseInstance(inst);
			for (int attr = 0; attr < initialInst.numAttributes(); attr++){
				initialInst.setValue(attr, 0);
			}
			this.centroids.add(initialInst);
			initialInst.setClassValue(1);
			this.centroids.add(initialInst);
		}

		// Removing the oldest instance in the window
		if (this.window.numInstances() >= this.limitOption.getValue()) {
			Instance removedInst = this.window.get(0);

			if (debugFlag.isSet()) System.out.println("window size before deletion: " + window.size());
			this.window.delete(0);
			if (debugFlag.isSet()) System.out.println("window size after deletion: " + window.size());
			// Remove the counters and sum
			int removedClassVal = (int)removedInst.classValue();
			Instance removedCentInst = this.centroids.get(removedClassVal);

			this.classCounters[removedClassVal]--;

			for (int attr = 0; attr < inst.numAttributes(); attr++) {
				this.attrSums[removedClassVal][attr] -= removedInst.value(attr);
				removedCentInst.setValue(attr, this.attrSums[removedClassVal][attr]);
			}
		}

		// Adding the instance to the appropriate centroid
		if (debugFlag.isSet()) System.out.println("Adding a new inst of class " + inst.classValue());
		this.window.add(inst);
		if (debugFlag.isSet()) System.out.println("window size after adding: " + window.size());
		int classVal = (int)inst.classValue();
		// add counter
		classCounters[classVal]++;
		if (debugFlag.isSet()) System.out.println("Class 0 count: " + classCounters[0] + "	Class 1 count: " + classCounters[1]);
		// add to sum and update the centroid
		Instance centInst = this.centroids.get(classVal);
		// Adding the instance, and all the attributes, to the sums array
		// Then recalculating the centroid
		for (int attr = 0; attr < inst.numAttributes(); attr++) {
			attrSums[classVal][attr] = attrSums[classVal][attr] + inst.value(attr);
			centInst.setValue(attr, attrSums[classVal][attr] / classCounters[classVal]);
		}
    }

	@Override
    public double[] getVotesForInstance(Instance inst) {
		double v[] = new double[C+1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				search = new LinearNNSearch(this.centroids);
			} else {
				search = new KDTree();
				search.setInstances(this.centroids);
			}	
			if (this.centroids.numInstances()>0) {
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.centroids.numInstances()));

				for (int i = 0; i < neighbours.numInstances(); i++) {
					v[(int) neighbours.instance(i).classValue()]++;
				}
			}
		} catch(Exception e) {
			return new double[inst.numClasses()];
		}
		return v;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}