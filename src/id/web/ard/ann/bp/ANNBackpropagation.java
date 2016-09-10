/*
 * Ardiansyah | http://ard.web.id
 */
package id.web.ard.ann.bp;

import java.util.Random;

/**
 *
 * @author Ardiansyah <ard333.ardiansyah@gmail.com>
 */
public final class ANNBackpropagation {
	
	private final Integer numOfInput;
	private final Integer numOfHidden;
	private final Integer numOfOutput;
	
	private final Double learningRate;
	private final Double minError;
	
	private Double[] X;//input
	private Double[] Z;//hidden
	private Double[] Y;//output
	
	private Double[][] v;//input->hidden
	private Double[][] w;//hidden->output
	
	private Double[][] deltaW;
	private Double[][] deltaV;
	
	private Double[][] inputTraining;
	private Double[][] expectedOutput;
	
	private Integer epoch;

	public ANNBackpropagation(Integer numOfInput, Integer numOfHidden, Integer numOfOutput, Double learningRate, Double minError) {
		this.numOfInput = numOfInput;
		this.numOfHidden = numOfHidden;
		this.numOfOutput = numOfOutput;
		this.learningRate = learningRate;
		this.minError = minError;
		
		this.init();
	}
	
	private void init() {
		this.epoch = 0;
		
		this.X = new Double[numOfInput+1];
		this.Z = new Double[numOfHidden+1];
		this.Y = new Double[numOfOutput];
		this.X[numOfInput] = 1.0;//bias at last index
		this.Z[numOfHidden] = 1.0;//bias at last index
		
		this.v = new Double[numOfInput+1][numOfHidden];
		this.w = new Double[numOfHidden+1][numOfOutput];
		this.deltaV = new Double[numOfInput+1][numOfHidden];
		this.deltaW = new Double[numOfHidden+1][numOfOutput];
		
		Random r = new Random();
		
		for (int i = 0; i < this.numOfInput+1; i++) {
			for (int j = 0; j < this.numOfHidden; j++) {
				this.v[i][j] = -1 + (1 - (-1)) * r.nextDouble();//-1:1
			}
		}
		for (int i = 0; i < numOfHidden+1; i++) {
			for (int j = 0; j < numOfOutput; j++) {
				this.w[i][j] = -1 + (1 - (-1)) * r.nextDouble();//-1:1
			}
		}
	}
	
	public void setTrainingData(Double[][] inputTraining, Double[][] expectedOutput) {
		this.inputTraining = inputTraining;
		this.expectedOutput = expectedOutput;
	}
	
	public void train() {
		Double[] eO = new Double[numOfOutput];
		if (this.inputTraining!=null && this.expectedOutput!=null) {
			System.out.println("Learning Process, please wait...");
			Double err = 0.0;
			do {
				this.epoch++;
				for (int i = 0; i < this.inputTraining.length; i++) {
					System.arraycopy(this.inputTraining[i], 0, X, 0, this.inputTraining[i].length);
					System.arraycopy(this.expectedOutput[i], 0, eO, 0, this.expectedOutput[i].length);
					
					this.feedForward();
					this.backPropagation(eO);
				}
				err = this.caclERR();
				System.out.println("Error: "+err);
			}while (err > this.minError);
		} else {
			System.out.println("Tidak ada data training...");
		}
	}
	private Double caclERR() {
		Double[] eO = new Double[numOfOutput];
		Double err = 0.0;
		Double errTotal = 0.0;
		
		for (int i = 0; i < this.inputTraining.length; i++) {
			System.arraycopy(this.inputTraining[i], 0, X, 0, this.inputTraining[i].length);
			System.arraycopy(this.expectedOutput[i], 0, eO, 0, this.expectedOutput[i].length);
			this.feedForward();
			for (int a = 0; a < this.numOfOutput; a++) {
				err += Math.pow((eO[a]-this.Y[a]),2);
			}
			err /= numOfOutput;
			errTotal += err;
		}
		errTotal /= this.inputTraining.length;
		return errTotal;
	}
	
	public void test(Double[] input) {
		System.arraycopy(input, 0, this.X, 0, this.numOfInput);
		this.feedForward();
	}
	
	private void feedForward() {
		this.setOutputZ();
		this.setOutputY();
	}
	private void setOutputZ() {
		Double temp[] = new Double[numOfHidden];
		for (int a = 0; a < numOfHidden; a++) {
			temp[a] = 0.0;
		}
		for (int j = 0; j < this.numOfHidden; j++) {
			for (int i = 0; i < this.numOfInput+1; i++) {
				temp[j] = temp[j] + this.X[i] * this.v[i][j];
			}
		}
		for (int j = 0; j < numOfHidden; j++) {
			this.Z[j] = this.sigmoid(temp[j]);
		}
	}
	private void setOutputY() {
		Double temp[] = new Double[numOfOutput];
		for (int a = 0; a < numOfOutput; a++) {
			temp[a] = 0.0;
		}
		for (int k = 0; k < this.numOfOutput; k++) {
			for (int j = 0; j < this.numOfHidden+1; j++) {
				temp[k] = temp[k] + this.Z[j] * this.w[j][k];
			}
		}
		for (int k = 0; k < this.numOfOutput; k++) {
			this.Y[k] = this.sigmoid(temp[k]);
		}
	}
	
	private void backPropagation(Double[] expectedOutput) {
		Double[] fO = new Double[this.numOfOutput];
		
		for (int k = 0; k < numOfOutput; k++) {
			fO[k] = (expectedOutput[k]-this.Y[k])*this.Y[k]*(1-this.Y[k]);
		}
		for (int j = 0; j < this.numOfHidden+1; j++) {//+bias weight
			for (int k = 0; k < this.numOfOutput; k++) {
				this.deltaW[j][k] = this.learningRate * fO[k] * this.Z[j];
			}
		}
		Double[] fHNet = new Double[this.numOfHidden];
		for (int j = 0; j < this.numOfHidden; j++) {
			fHNet[j] = 0.0;
			for (int k = 0; k < this.numOfOutput; k++) {
				fHNet[j] = fHNet[j] + (fO[k]*this.w[j][k]);
			}
		}
		Double[] fH = new Double[this.numOfHidden];
		for (int j = 0; j < this.numOfHidden; j++) {
			fH[j] = fHNet[j]*this.Z[j]*(1-this.Z[j]);
		}
		for (int i = 0; i < this.numOfInput+1; i++) {
			for (int j = 0; j < numOfHidden; j++) {
				this.deltaV[i][j] = this.learningRate * fH[j] * this.X[i];
			}
		}
		this.changeWeight();
	}
	private void changeWeight() {
		for (int j = 0; j < numOfHidden+1; j++) {
			for (int k = 0; k < numOfOutput; k++) {
				this.w[j][k] = this.w[j][k] + this.deltaW[j][k];
			}
		}
		for (int i = 0; i < numOfInput+1; i++) {
			for (int j = 0; j < numOfHidden; j++) {
				this.v[i][j] = this.v[i][j] + this.deltaV[i][j];
			}
		}
	}
	
	private Double sigmoid(Double input) {
		return 1 / (1 + (double)Math.exp(-input));
	}
	public Double[] getOutput() {
		return this.Y;
	}
	public Integer getEpoch() {
		return this.epoch;
	}
}
