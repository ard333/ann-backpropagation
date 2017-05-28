/*
 * Ardiansyah | http://ard.web.id
 */
package id.web.ard.ann.bp;

/**
 *
 * @author Ardiansyah <ard333.ardiansyah@gmail.com>
 */
public class Main {

	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) {
		
		ANNBackpropagation annBPSigmoid = new ANNBackpropagation(2, 3, 1, 0.5, 0.01, 10000, ActivationFunction.SIGMOID);

		//==========TRAIN==========
		Double[][] patternInput = new Double[4][2];
		Double[][] expectedOutput = new Double[4][1];
		
		patternInput[0][0] = 0.0;	patternInput[0][1] = 0.0;	expectedOutput[0][0] = 0.0;
		patternInput[1][0] = 1.0;	patternInput[1][1] = 1.0;	expectedOutput[1][0] = 0.0;
		patternInput[2][0] = 1.0;	patternInput[2][1] = 0.0;	expectedOutput[2][0] = 1.0;
		patternInput[3][0] = 0.0;	patternInput[3][1] = 1.0;	expectedOutput[3][0] = 1.0;
		
		annBPSigmoid.setTrainingData(patternInput, expectedOutput);
		annBPSigmoid.train();
		System.out.println("Epoch : "+annBPSigmoid.getEpoch());
		//=========================
		
		//==========TEST===========
		Double[] dataTest = new Double[2];
		Double[] output;
		
		dataTest[0] = 0.0;dataTest[1] = 0.0;
		annBPSigmoid.test(dataTest);
		output = annBPSigmoid.getOutput();
		
		System.out.println("0 0 -> "+output[0]);
		
		
		dataTest[0] = 0.0;dataTest[1] = 1.0;
		annBPSigmoid.test(dataTest);
		
		output = annBPSigmoid.getOutput();
		System.out.println("0 1 -> "+output[0]);
		
		
		dataTest[0] = 1.0;dataTest[1] = 0.0;
		annBPSigmoid.test(dataTest);
		
		output = annBPSigmoid.getOutput();
		System.out.println("1 0 -> "+output[0]);
		
		
		dataTest[0] = 1.0;dataTest[1] = 1.0;
		annBPSigmoid.test(dataTest);
		
		output = annBPSigmoid.getOutput();
		System.out.println("1 1 -> "+output[0]);
		//=====================
		
	}
		
}
