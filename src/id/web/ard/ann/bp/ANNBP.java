/*
 * Ardiansyah | http://ard.web.id
 */
package id.web.ard.ann.bp;

/**
 *
 * @author Ardiansyah <ard333.ardiansyah@gmail.com>
 */
public class ANNBP {

	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) {
		
		ANNBackpropagation annBP = new ANNBackpropagation(2, 3, 1, 0.8, 0.001);
		
		//==========TRAIN==========
		Double[][] patternInput = new Double[4][2];
		Double[][] expectedOutput = new Double[4][1];
		
		patternInput[0][0] = 0.0;	patternInput[0][1] = 0.0;	expectedOutput[0][0] = 0.0;
		patternInput[1][0] = 1.0;	patternInput[1][1] = 1.0;	expectedOutput[1][0] = 0.0;
		patternInput[2][0] = 1.0;	patternInput[2][1] = 0.0;	expectedOutput[2][0] = 1.0;
		patternInput[3][0] = 0.0;	patternInput[3][1] = 1.0;	expectedOutput[3][0] = 1.0;
		
		annBP.setTrainingData(patternInput, expectedOutput);
		annBP.train();
		System.out.println("Epoch : "+annBP.getEpoch());
		//=========================
		
		//==========TEST===========
		Double[] dataTest = new Double[2];
		Double[] output;
		
		dataTest[0] = 0.0;dataTest[1] = 0.0;
		annBP.test(dataTest);
		output = annBP.getOutput();
		for (Double output1 : output) {
			System.out.println("0 xor 0 -> "+output1);
		}
		
		dataTest[0] = 0.0;dataTest[1] = 1.0;
		annBP.test(dataTest);
		output = annBP.getOutput();
		for (Double output1 : output) {
			System.out.println("0 xor 1 -> "+output1);
		}
		
		dataTest[0] = 1.0;dataTest[1] = 0.0;
		annBP.test(dataTest);
		output = annBP.getOutput();
		for (Double output1 : output) {
			System.out.println("1 xor 0 -> "+output1);
		}
		
		dataTest[0] = 1.0;dataTest[1] = 1.0;
		annBP.test(dataTest);
		output = annBP.getOutput();
		for (Double output1 : output) {
			System.out.println("1 xor 1 -> "+output1);
		}
		//=====================
	}
}
