import java.util.Arrays;
import java.util.Random;

public class MLP
{
    private int numInputs;
    private int numHiddenUnits;
    private int numOutputs;
    //used to store network architecture parameters

    private double[][] w1, w2;
    private double[][] dW1, dW2;
    //store weights and weight changes for input layer and hidden layer respectively

    private double[] z1, z2;
    //store activations (weight * input before applying sigmoid) to hidden units and output respectively

    private double[] inputValues;//need input values when computing deltas for lower layer

    private double[] hidden;
    private double[] outputs;
    //store hidden and output values after sigmoid activation function is applied to the respective input


    public MLP(int ni, int nh, int no)
    {

        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;
        //keeping track of input parameters for network architecture

        w1 = new double [numInputs][numHiddenUnits];
        w2 = new double [numHiddenUnits][numOutputs];

        dW1 = new double [numInputs][numHiddenUnits];
        dW2 = new double [numHiddenUnits][numOutputs];

        z1 = new double[numHiddenUnits];
        z2 = new double[numOutputs];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //create various arrays of appropriate size

        w1 = randomise(w1);
        w2 = randomise(w2);
        Arrays.fill(z1,0);
        Arrays.fill(z2, 0);
        Arrays.fill(hidden,0);
        Arrays.fill(outputs, 0);
        //fill weight arrays with small random initial weights

        dW1 = fill2Dzeroes(dW1);
        dW2 = fill2Dzeroes(dW2);
        //fill weight change arrays with 0's intially so can use += later
    }

    private double[][] randomise(double[][] input)
    //method to fill 2d array with small random values
    {
        Random rand = new Random();

        double[][] result = new double[input.length][input[0].length];//output array same dimension as input array

        for(int i=0;i<input.length;i++)
        {
            for(int j=0;j<input[0].length;j++)
                result[i][j] = rand.nextDouble();//fill array with random doubles in range [0,1)
        }

        return result;
    }

    private double[][] fill2Dzeroes(double[][] input)//method to fill 2D array with 0's - need for dW1 & dW2 reset in update weights method
    {
        double[][] result = new double[input.length][input[0].length];

        for(double[] row : result)
            Arrays.fill(row,0);

        return result;//array of same dimension as input and filled with 0's
    }

    public void forwardPass(double[] input)//pass an input vector and calculate and store result in outputs array
    {
        inputValues = input;//keep track of input data - need this to calculate deltas
        calculateHidden();
        calculateOutput();
    }

    private void calculateHidden()//calculate hidden values for forward pass
    {

        for(int i=0;i<numHiddenUnits;i++)//using matrix multiplication to calculate values in hidden units (input * w1)
        {
            for(int j=0;j<numInputs;j++)
            {
                z1[i] = (inputValues[j] * w1[j][i]);//activation for hidden units
                hidden[i] = sigmoid(z1[i]);//apply activation to get hidden values
            }
        }
    }

    private void calculateOutput()//calculate output of a forward pass
    {
        for(int i=0;i<numOutputs;i++)//multiply input to layer (hidden values) by weights for hidden layer (hidden * w2)
        {
            for(int j=0;j<numInputs;j++)
            {
                z2[i] =z1[j] * w2[j][i];//activation for output
                outputs[i] = sigmoid(z2[i]);//apply activation to get output values
            }
        }
    }

    public double backProp(double[] target)//back propagate error and calculate weight changes - apply chain rule
    {
        double deltaOutput = 0;//need to initialise a value (but calculate it in below for loop)

        for(int i=0;i<numHiddenUnits;i++)//need to compute deltas for upper layer
        {
            for(int j=0;j<numOutputs;j++)
            {
                deltaOutput = calculateError(target[j], outputs[j]) * sigmoidDerivative(z2[j]);
                dW2[i][j] += deltaOutput * hidden[j]; //multiply deltas by hidden values to get dW2 values
            }
        }

        for(int i=0;i<numInputs;i++)//compute deltas for lower layer
        {
            for(int j=0;j<numHiddenUnits;j++)
            {
                double hiddenError = deltaOutput * sigmoidDerivative(hidden[j]);
                double deltaHidden = hiddenError * w2[j][0];
                dW1[i][j] +=  deltaHidden * inputValues[i]; //multiply deltas by input values to get dW1 values
            }
        }

        return calculateTotalError(target);//return total error for a given example input
    }

    private double calculateTotalError(double[] target)
    //calculates total error over multiple target values (XOR and sin questions only have 1)
    {
        double totalError = 0;

        for(int i=0;i<numOutputs;i++)
            totalError += calculateError(target[i], outputs[i]);//summation of error

        return totalError;
    }

    private double calculateError(double target, double actual)//calculate error from target value
    {
        return  0.5*(Math.pow((target-actual),2));//mean squared error
    }

    public void updateWeights(double learningRate)//apply weight updates with learning rate
    {
        for(int i=0;i<numInputs;i++)
        {
            for(int j=0;j<numHiddenUnits;j++)
                w1[i][j] -=dW1[i][j]*learningRate;//- value for learning rate
        }

        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numOutputs;j++)
                w2[i][j] -= dW2[i][j]*learningRate;//- value for learning rate
        }

        dW1 = fill2Dzeroes(dW1);
        dW2 = fill2Dzeroes(dW2);
        //reset the weight change arrays to 0
    }

    private double sigmoid(double x)//activation function
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)//activation function derivative for backpropagation
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }

    public double[] getOutput()//get output value(s)
    {
        return outputs;
    }
}
