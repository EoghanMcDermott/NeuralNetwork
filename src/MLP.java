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

    private double[] hidden;

    private double[] outputs;



    public MLP(int ni, int nh, int no)
    {

        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;

        w1 = new double [numInputs][numHiddenUnits];
        w2 = new double [numHiddenUnits][numOutputs];

        z1 = new double[numHiddenUnits];
        z2 = new double[numHiddenUnits];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //initialise various arrays

        w1 = randomise(w1);
        w2 = randomise(w2);
        Arrays.fill(hidden,0);
        Arrays.fill(outputs, 0);
        //fill weight arrays with small random initial weights
    }

    private double[][] randomise(double[][] input)//method to fill 2d array with small random values
    {
        double[][] result = new double[input.length][input[0].length];

        for(int i=0;i<input.length;i++)
        {
            for(int j=0;j<input[0].length;j++)
                result[i][j] = generateInitialWeight();
        }

        return result;
    }

    private double generateInitialWeight()//method to generate small initial weights
    {
        Random rand = new Random();
//
//        int numerator = rand.nextInt(10);
//
//        double initialWeight = numerator/1000.0;
//
//        if(initialWeight == 0)
//            initialWeight += 0.05;
//
//        return initialWeight;
        return rand.nextDouble();
    }

    public void forwardPass(double[] input)
    {
        calculateHidden(input);
        calculateOutput();
    }

    public void backProp(double target)
    {

    }

    public void updateWeights(double learningRate)
    {
        //W1 += learningRate*dW1;
       // W2 += learningRate*dW2;

        Arrays.fill(dW1,0);
        Arrays.fill(dW2,0);
    }

    private double sigmoid(double x)//activation function
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)//activation function derivative for backpropagation
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }

    private void calculateHidden(double[] input)
    {
        System.out.println("w1 values:");
        for(double[] row : w1)
            for(double w : row)
                System.out.println(w);

        //using matrix multiplication to calculate values in hidden units
        for(int i=0;i<numHiddenUnits;i++)
        {
            for(int j=0;j<numInputs;j++)
                hidden[i] += input[j] * w1[j][i];
        }

        System.out.println("Hidden before activation:");
        for(double h : hidden)
            System.out.println("Hidden: " + h);

        for(int i=0;i<numHiddenUnits;i++)
        {
            hidden[i] = sigmoid(hidden[i]);//applying activation function
            z1[i] = hidden[i];
        }
    }

    private void calculateOutput()
    {
       //w2 = new double[][]{{0.3},{0.5}};

        for(int i=0;i<numOutputs;i++)
        {
            for(int j=0;j<numInputs;j++)
                outputs[i] += hidden[j] * w2[j][i];
        }

        System.out.println("Output: " + outputs[0]);
        for(double o: outputs)//applying activation function
            o = sigmoid(o);

        System.out.println("Result after activation: " + outputs[0]);

    }
}
