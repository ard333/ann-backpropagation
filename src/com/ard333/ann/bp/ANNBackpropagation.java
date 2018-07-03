package com.ard333.ann.bp;

import java.util.ArrayList;
import java.util.Random;

/**
 * Artificial Neural Network with Backpropagation Training Algorithm.
 * 
 * @author Ardiansyah <ard333.ardiansyah@gmail.com>
 */
public final class ANNBackpropagation {
	
	private final Integer numOfInput;
	private final Integer numOfHidden;
	private final Integer numOfOutput;
	
	private final Double learningRate;
	private final Double minError;
	
	private Double[] I;//input
	private Double[] H1;//hidden
	private Double[] O;//output
	
	private Double[][] w1;//input->hidden
	private Double[][] w2;//hidden->output
	
	private Double[] sigmaForH1;
	private Double[] sigmaForO;
	
	private Double[][] deltaw1;
	private Double[][] deltaw2;
	
	private Double[][] inputTraining;
	private Double[][] expectedOutput;
	
	private Integer epoch;
	private Integer maxEpoch;
	private ActivationFunction activationFunction;
	
	/**
	 * Create new Artificial Neural Network with specify parameters.
	 * 
	 * @param numOfInput number of input unit.
	 * @param numOfHidden number of hidden neuron.
	 * @param numOfOutput number of output neuron.
	 * @param learningRate learning rate (0.1 - 1).
	 * @param minError minimal error.
	 * @param maxEpoch maximum training iteration.
	 * @param activationFunction selected activation function.
	 */
	public ANNBackpropagation(
			Integer numOfInput, Integer numOfHidden, Integer numOfOutput, Double learningRate, Double minError,
			Integer maxEpoch, ActivationFunction activationFunction
	) {
		this.numOfInput = numOfInput;
		this.numOfHidden = numOfHidden;
		this.numOfOutput = numOfOutput;
		
		this.learningRate = learningRate;
		this.minError = minError;
		this.maxEpoch = maxEpoch;
		
		this.activationFunction = activationFunction;
		this.init();
	}
	
	/**
	 * Initialize arrays and give random weights.
	 */
	private void init() {
		this.epoch = 0;
		
		this.I = new Double[numOfInput+1];
		this.H1 = new Double[numOfHidden+1];
		this.O = new Double[numOfOutput];
		this.I[numOfInput] = 1.0;//bias at last index
		this.H1[numOfHidden] = 1.0;//bias at last index
		
		this.sigmaForH1 = new Double[numOfHidden+1];
		this.sigmaForO = new Double[numOfOutput];
		
		this.w1 = new Double[numOfInput+1][numOfHidden];
		this.w2 = new Double[numOfHidden+1][numOfOutput];
		this.deltaw1 = new Double[numOfInput+1][numOfHidden];
		this.deltaw2 = new Double[numOfHidden+1][numOfOutput];
		
		Random r = new Random();
		
		for (int i = 0; i < this.numOfInput+1; i++) {
			for (int j = 0; j < this.numOfHidden; j++) {
				this.w1[i][j] = -1 + (1 - (-1)) * r.nextDouble();//-1:1
			}
		}
		for (int i = 0; i < numOfHidden+1; i++) {
			for (int j = 0; j < numOfOutput; j++) {
				this.w2[i][j] = -1 + (1 - (-1)) * r.nextDouble();//-1:1
			}
		}
		this.nguyenWidrow(w1, this.numOfInput, this.numOfHidden);
		this.nguyenWidrow(w2, this.numOfHidden, this.numOfOutput);
	}
	
	/**
	 * Nguyen Widrow intilialize
	 * @param input weights between two layer
	 * @param n number of first layer.
	 * @param p number of second layer.
	 */
	private void nguyenWidrow(Double[][] input, int n, int p) {
		double beta = 0.7 * Math.pow(p, 1.0 / n);
		double[] v = new double[p];
		for (int j = 0; j < p; j++) {
			double temp = 0.0;
			for (int x = 0; x < n; x++) {
				temp += input[x][j] * input[x][j];
			}
			v[j] = Math.sqrt(Math.abs(temp));
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < p; j++) {
				input[i][j] = beta * input[i][j] / v[j];
			}
		}
		for (int i = 0; i < p; i++) {
			input[n][p-1] = -beta + (beta - (-beta)) * new Random().nextDouble();
		}
	}
	
	/**
	 * Set each pattern (Training Data) and Expected Output.
	 * 
	 * @param inputTraining set of training data.
	 * @param expectedOutput set of expected output.
	 */
	public void setTrainingData(Double[][] inputTraining, Double[][] expectedOutput) {
		this.inputTraining = inputTraining;
		this.expectedOutput = expectedOutput;
	}
	
	/**
	 * Train ANN until error minimum reached.
	 */
	public void train() {
		Double[] eO = new Double[numOfOutput];
		if (this.inputTraining!=null && this.expectedOutput!=null) {
			System.out.println("Learning Process, please wait...");
			Double err = 0.0;
			Random r = new Random();
			do {
				this.epoch++;
				int j;
				for (int i = 0; i < this.inputTraining.length; i++) {
					j = r.nextInt(inputTraining.length);
					System.arraycopy(this.inputTraining[j], 0, I, 0, this.inputTraining[j].length);
					System.arraycopy(this.expectedOutput[j], 0, eO, 0, this.expectedOutput[j].length);
					
					this.feedForward();
					this.backPropagation(eO);
				}
				
				err = this.caclERR();
				System.out.println("Error: "+err);
			}while (err > this.minError && this.epoch < this.maxEpoch);
		} else {
			System.out.println("No training data...");
		}
	}
	
	/**
	 * Calculate error average for all pattern.
	 * 
	 * @return error average.
	 */
	private Double caclERR() {
		Double[] eO = new Double[numOfOutput];
		Double err;
		Double errTotal = 0.0;
		
		for (int i = 0; i < this.inputTraining.length; i++) {
			err = 0.0;
			System.arraycopy(this.inputTraining[i], 0, I, 0, this.inputTraining[i].length);
			System.arraycopy(this.expectedOutput[i], 0, eO, 0, this.expectedOutput[i].length);
			this.feedForward();
			for (int a = 0; a < this.numOfOutput; a++) {
				err += Math.pow((eO[a]-this.O[a]),2);
			}
			err /= numOfOutput;
			errTotal += err;
		}
		errTotal /= this.inputTraining.length;
		return errTotal;
	}
	
	/**
	 * Test pattern after training.
	 * 
	 * @param input input pattern.
	 */
	public void test(Double[] input) {
		System.arraycopy(input, 0, this.I, 0, this.numOfInput);
		this.feedForward();
	}
	
	/**
	 * Feed-forward.
	 */
	private void feedForward() {
		this.setOutputH1();
		this.setOutputO();
	}
	
	/**
	 * Calculate each output of hidden neuron.
	 */
	private void setOutputH1() {
		for (int a = 0; a < numOfHidden; a++) {
			this.sigmaForH1[a] = 0.0;
		}
		for (int j = 0; j < this.numOfHidden; j++) {
			for (int i = 0; i < this.numOfInput+1; i++) {
				this.sigmaForH1[j] = this.sigmaForH1[j] + this.I[i] * this.w1[i][j];
			}
		}
		for (int j = 0; j < numOfHidden; j++) {
			if (null != this.activationFunction) switch (this.activationFunction) {
				case SIGMOID:
					this.H1[j] = this.sigmoid(this.sigmaForH1[j]);
					break;
				case BIPOLAR_SIGMOID:
					this.H1[j] = this.bipolarSigmoid(this.sigmaForH1[j]);
					break;
				case TANH:
					this.H1[j] = this.tanH(this.sigmaForH1[j]);
					break;
				default:
					break;
			}
		}
	}
	
	/**
	 * Calculate each output of output neuron.
	 */
	private void setOutputO() {
		for (int a = 0; a < numOfOutput; a++) {
			this.sigmaForO[a] = 0.0;
		}
		for (int k = 0; k < this.numOfOutput; k++) {
			for (int j = 0; j < this.numOfHidden+1; j++) {
				this.sigmaForO[k] = this.sigmaForO[k] + this.H1[j] * this.w2[j][k];
			}
		}
		for (int k = 0; k < this.numOfOutput; k++) {
			if (null != this.activationFunction) switch (this.activationFunction) {
				case SIGMOID:
					this.O[k] = this.sigmoid(this.sigmaForO[k]);
					break;
				case BIPOLAR_SIGMOID:
					this.O[k] = this.bipolarSigmoid(this.sigmaForO[k]);
					break;
				case TANH:
					this.O[k] = this.tanH(this.sigmaForO[k]);
					break;
				default:
					break;
			}
		}
	}
	
	/**
	 * Backpropagation.
	 * 
	 * @param expectedOutput set of expected output.
	 */
	private void backPropagation(Double[] expectedOutput) {
		Double[] fO = new Double[this.numOfOutput];
		
		for (int k = 0; k < numOfOutput; k++) {
			if (null != this.activationFunction) switch (this.activationFunction) {
				case SIGMOID:
					fO[k] = (expectedOutput[k]-this.O[k]) * this.sigmoidDerivative(this.sigmaForO[k]);
					break;
				case BIPOLAR_SIGMOID:
					fO[k] = (expectedOutput[k]-this.O[k]) * this.bipolarSigmoidDerivative(this.sigmaForO[k]);
					break;
				case TANH:
					fO[k] = (expectedOutput[k]-this.O[k]) * this.tanHDerivative(this.sigmaForO[k]);
					break;
				default:
					break;
			}
		}
		for (int j = 0; j < this.numOfHidden+1; j++) {//+bias weight
			for (int k = 0; k < this.numOfOutput; k++) {
				this.deltaw2[j][k] = learningRate * fO[k] * this.H1[j];
			}
		}
		Double[] fHNet = new Double[this.numOfHidden];
		for (int j = 0; j < this.numOfHidden; j++) {
			fHNet[j] = 0.0;
			for (int k = 0; k < this.numOfOutput; k++) {
				fHNet[j] = fHNet[j] + (fO[k]*this.w2[j][k]);
			}
		}
		Double[] fH = new Double[this.numOfHidden];
		for (int j = 0; j < this.numOfHidden; j++) {
			if (null != this.activationFunction) switch (this.activationFunction) {
				case SIGMOID:
					fH[j] = fHNet[j] * this.sigmoidDerivative(this.sigmaForH1[j]);
					break;
				case BIPOLAR_SIGMOID:
					fH[j] = fHNet[j] * this.bipolarSigmoidDerivative(this.sigmaForH1[j]);
					break;
				case TANH:
					fH[j] = fHNet[j] * this.tanHDerivative(this.sigmaForH1[j]);
					break;
				default:
					break;
			}
		}
		for (int i = 0; i < this.numOfInput+1; i++) {
			for (int j = 0; j < numOfHidden; j++) {
				this.deltaw1[i][j] = learningRate * fH[j] * this.I[i];
			}
		}
		this.changeWeight();
	}
	
	/**
	 * Update all weights.
	 */
	private void changeWeight() {
		for (int j = 0; j < numOfHidden+1; j++) {
			for (int k = 0; k < numOfOutput; k++) {
				this.w2[j][k] = this.w2[j][k] + this.deltaw2[j][k];
			}
		}
		for (int i = 0; i < numOfInput+1; i++) {
			for (int j = 0; j < numOfHidden; j++) {
				this.w1[i][j] = this.w1[i][j] + this.deltaw1[i][j];
			}
		}
	}
	
	/**
	 * Sigmoid Activation Function.
	 * <br/>f(x) = 1 / (1 + exp(-x))
	 * 
	 * @param x an input value.
	 * @return a result of Sigmoid Activation Function.
	 */
	private Double sigmoid(Double x) {
		return 1 / (1 + (double)Math.exp(-x));
	}
	
	/**
	 * Derivative of Sigmoid Activation Function.
	 * <br/>f'(x) = f(x) * (1 - f(x))
	 * 
	 * @param x an input value.
	 * @return  a result of Derivative Sigmoid Activation Function.
	 */
	private Double sigmoidDerivative(Double x) {
		return this.sigmoid(x) * (1-this.sigmoid(x));
	}
	
	/**
	 * Sigmoid Bipolar Activation Function.
	 * <br/>f(x) = 2 / (1 + exp(-x)) - 1
	 * 
	 * @param x an input value.
	 * @return a result of Sigmoid Bipolar Activation Function.
	 */
	private Double bipolarSigmoid(Double x) {
		return 2/(1+Math.exp(-x))-1;
	}
	
	/**
	 * Derivative of Sigmoid Bipolar Activation Function.
	 * <br/>f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))
	 * 
	 * @param x an input value.
	 * @return  a result of Derivative Sigmoid Bipolar Activation Function.
	 */
	private Double bipolarSigmoidDerivative(Double x) {
		return 0.5 * (1+this.bipolarSigmoid(x)) * (1-this.bipolarSigmoid(x));
	}
	
	/**
	 * TanH Activation Function.
	 * <br/>f(x) = 2 / (1 + exp(-x)) - 1
	 * <br/>output range -1 until 1.
	 * 
	 * @param x an input value.
	 * @return a result of TanH Activation Function.
	 */
	private Double tanH(Double x) {
		return 2/(1 + Math.exp(-2*x))-1;
	}
	
	/**
	 * Derivative of TanH Activation Function.
	 * <br/>f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))
	 * <br/>output range -1 until 1.
	 * 
	 * @param x an input value.
	 * @return  a result of Derivative TanH Activation Function.
	 */
	private Double tanHDerivative(Double x) {
		return 1- Math.pow(this.tanH(x), 2);
	}
	
	/**
	 * Method for getting output of each output neuron.
	 * 
	 * @return output of each output neuron.
	 */
	public Double[] getOutput() {
		return this.O;
	}
	
	/**
	 * Method for getting epoch until minimum error reached.
	 * 
	 * @return epoch until minimum error reached. 
	 */
	public Integer getEpoch() {
		return this.epoch;
	}
	
}
