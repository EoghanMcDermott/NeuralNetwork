import java.util.Arrays;
import java.util.Random;

public class MLP
{
    private int numInputs;
    private int numHiddenUnits;
    private int numOutputs;

    private double[][] w1, w2;
    private double[][] dW1, dW2;

    private double[] z1, z2;

    private double[] inputValues;
    private double[] hidden;
    private double[] outputs;



    public MLP(int ni, int nh, int no)
    {

        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;

        w1 = new double [numInputs][numHiddenUnits];
        w2 = new double [numHiddenUnits][numOutputs];

        dW1 = new double [numInputs][numHiddenUnits];
        dW2 = new double [numHiddenUnits][numOutputs];

        z1 = new double[numHiddenUnits];
        z2 = new double[numOutputs];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //initialise various arrays

        w1 = randomise(w1);
        w2 = randomise(w2);
        Arrays.fill(z1,0);
        Arrays.fill(z2, 0);
        Arrays.fill(hidden,0);
        Arrays.fill(outputs, 0);
        //fill weight arrays with small random initial weights
    }

    private double[][] fill2Dzeroes(double[][] input)
    {
        double[][] result = new double[input.length][input[0].length];

        for(double[] row : result)
            Arrays.fill(row,0);

        return result;
    }

    private double[][] randomise(double[][] input)//method to fill 2d array with small random values
    {
        Random rand = new Random();

        double[][] result = new double[input.length][input[0].length];

        for(int i=0;i<input.length;i++)
        {
            for(int j=0;j<input[0].length;j++)
                result[i][j] = rand.nextDouble();
        }

        return result;
    }

    public void forwardPass(double[] input)
    {
        inputValues = input;
        calculateHidden(input);
        calculateOutput();
    }

    private void calculateHidden(double[] input)
    {
        //using matrix multiplication to calculate values in hidden units
        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numInputs;j++)
            {
                hidden[i] = sigmoid(input[j] * w1[j][i]);
                z1[i] = hidden[i];
            }
        }
    }

    private void calculateOutput()
    {
        for(int i=0;i<numOutputs;i++)
        {
            for(int j=0;j<numInputs;j++)
            {
                outputs[i] = sigmoid(z1[j] * w2[j][i]);
                z2[i] = outputs[i];
            }
        }

        printOutput();
    }

    public double backProp(double target)
    {
        double error = calculateError(target);
        double deltaOutput = 0;

        for(int i=0;i<numOutputs;i++)
            deltaOutput += error * sigmoidDerivative(z2[i]);
        //need to compute deltas for upper layer

        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numOutputs;j++)
            {
                dW2[i][j] += deltaOutput * hidden[j];
            }
            //multiply deltas by hidden values
            //this gives dW2 values
        }
        double deltaHidden = 0;
        for(int i=0;i<numInputs;i++)
        {
            for(int j=0;j<numHiddenUnits;j++)
            {
                deltaHidden += deltaOutput * w1[i][j] * sigmoidDerivative(z1[j]);
                dW1[i][j] +=  deltaHidden * inputValues[i];
            }
        }
        //compute deltas for lower layer
        //multiply deltas by input values

        return error;
    }

    public void updateWeights(double learningRate)
    {
        for(int i=0;i<numInputs;i++)
        {
            for(int j=0;j<numHiddenUnits;j++)
                w1[i][j] -=dW1[i][j]*learningRate;
        }

        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numOutputs;j++)
                w2[i][j] -= dW2[i][j]*learningRate;
        }

       dW1 = fill2Dzeroes(dW1);
       dW2 = fill2Dzeroes(dW2);
    }

    private double sigmoid(double x)//activation function
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)//activation function derivative for backpropagation
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }



    private double calculateError(double target)
    {
        double error = 0;

        for(double y : outputs)
            error += Math.abs(y-target);
//            error+= 0.5*(Math.pow((y-target),2));

        return error;
    }

    private void printHidden()
    {
        for(double h : hidden)
            System.out.println("Hidden: " + h);
    }

    private void printOutput()
    {
        for(double o : outputs)
            System.out.println("Output: " + o);
    }

    private void printZ1()
    {
        for(double z : z1)
            System.out.println("Z1: " + z);
    }

    private void printZ2()
    {
        for(double z : z2)
            System.out.println("Output after activation: " + z);
    }

    private void printWeights()
    {
        System.out.println("w1:");
        for(double[] row : w1)
            for(double w : row)
                System.out.println(w);

        System.out.println("w2:");
        for(double[] column : w2)
            for(double w : column)
                System.out.println(w);
    }
}
