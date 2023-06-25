/*
 *    OzaBag.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
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
package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.Random;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiﬁer to train</li>
 * <li>-s : The number of models in the bag</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class TheEnsemble extends AbstractClassifier implements MultiClassClassifier,
                                                          CapabilitiesHandler {

    private int windowCount = 0;
    private Integer[] ensemblePerformance;
    private Integer[] instPerEnsembleMember;
    private int newLearnerPerformance = 0;
    private int instForNewLearner = 0;
    private Integer[] gracePeriod = {10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200};
    private double[] splitConfidence = {0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.00};
    private double[] tieThreshold = {0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.00};
    private Classifier newLearner;

    @Override
    public String getPurposeString() {
        return "Don't bother changing from Hoeffding Tree, it won't work";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, 100);

    public IntOption windowSize = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    public IntOption randomSeedOption = new IntOption("randomSeed", 'r',
            "Set the random seed.", 10, 1, 100);

    protected Classifier[] ensemble;

    @Override
    public void resetLearningImpl() {
        this.classifierRandom.setSeed(randomSeedOption.getValue());
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        this.ensemblePerformance = new Integer[ensemble.length];
        this.instPerEnsembleMember = new Integer[ensemble.length];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
            this.ensemblePerformance[i] = 0;
            this.instPerEnsembleMember[i] = 0;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        // New Learner every [Window Size] number of instances
        if (windowCount == 0){
            this.newLearner = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
            induceDiversity();
            newLearnerPerformance = 0;
            instForNewLearner = 0;
        }
        // Train all the current ensembles
        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
                if(this.ensemble[i].correctlyClassifies(inst)){
                    this.ensemblePerformance[i]++;
                }
                this.instPerEnsembleMember[i]++;
            }
        }
        // Train the learner on the instance
        int k = MiscUtils.poisson(1.0, this.classifierRandom);
        if (k > 0) {
            Instance weightedInst = (Instance) inst.copy();
            weightedInst.setWeight(inst.weight() * k);
            this.newLearner.trainOnInstance(weightedInst);
            if (this.newLearner.correctlyClassifies(inst)) {
                this.newLearnerPerformance++;
            }
            this.instForNewLearner++;
        }
        this.windowCount++;
        // Replace a learner in the ensemble with the New Learner or ignore New Learner
        // Reset the count so that a brand new learner is used for the next window size worth of instances
        if (this.windowCount >= this.windowSize.getValue()){
            replaceWorstClassifier();
            this.windowCount = 0;
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        double[] votes;
        double max = 0.0;
        int maxIndex = 0;
        double sum = 0.0;
        for (int i = 0; i < this.ensemble.length; i++) {
            votes = this.ensemble[i].getVotesForInstance(inst);
            max = 0.0;
            maxIndex = 0;
            sum = 0.0;
            for (int v = 0; v < votes.length; v++){
                sum += votes[v];
                if (max < votes[v]) {
                    max = votes[v];
                    maxIndex = v;
                }
            }
            if (sum > 0.0){
                combinedVote.addValues(new DoubleVector(createVote(maxIndex, votes.length, i)));
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                    this.ensemble != null ? this.ensemble.length : 0)};
    }

    private void induceDiversity () {
        ((HoeffdingTree) this.newLearner).gracePeriodOption.setValue(gracePeriod[this.classifierRandom.nextInt(20)]);
        ((HoeffdingTree) this.newLearner).splitConfidenceOption.setValue(splitConfidence[this.classifierRandom.nextInt(20)]);
        ((HoeffdingTree) this.newLearner).tieThresholdOption.setValue(tieThreshold[this.classifierRandom.nextInt(20)]);
    }

    private void replaceWorstClassifier () {
        int removedMember = 0;
        // Start the worst prediction off with the new learner so that if it changes then I know there is a worse member
        double worstPred = (double)this.newLearnerPerformance/this.instForNewLearner;
        double currPred = 0;
        for (int i = 0; i < this.ensemblePerformance.length; i++) {
            currPred = (double) this.ensemblePerformance[i]/this.instPerEnsembleMember[i];
            if (currPred < worstPred) {
                worstPred = currPred;
                removedMember = i;
            }
        }
        if (worstPred < currPred) {
            this.ensemble[removedMember] = this.newLearner;
            this.ensemblePerformance[removedMember] = newLearnerPerformance;
            this.instPerEnsembleMember[removedMember] = instForNewLearner;
        }
    }

    private double[] createVote(int index, int length, int ensemble) {
        double[] vote = new double[length];
        for (int i = 0; i < length; i++){
            if (i == index){
                vote[i] = this.ensemblePerformance[ensemble];
            } else {
                vote[i] = 0.0;
            }
        }
        return vote;
    }
}
