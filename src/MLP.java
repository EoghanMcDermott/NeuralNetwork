import java.util.Random;

public class MLP
{
    private int numInputs;
    private int numHiddenUnits;
    private int numOutputs;
    private double[][] w1, w2;
    private double[][] dW1, dW2;
    private double[][] z1, z2;
    private double[] hidden;
    private double[] outputs;



    public MLP(int ni, int nh, int no)
    {
        numInputs = ni;
        numHiddenUnits = nh;
        numOutputs = no;

        w1 = new double[numInputs][numHiddenUnits];
        w2 = new double[numOutputs][numHiddenUnits];

        z1 = new double[numHiddenUnits][1];
        z2 = new double[numHiddenUnits][1];

        hidden = new double[numHiddenUnits];
        outputs = new double[numOutputs];
        //initialise various arrays

        randomise(w1);
        randomise(w2);
        //fill weight arrays with small random initial weights
    }

    private void randomise(double[][] input)//method to fill 2d array with small random values
    {
        for(double[] w: input)
            for(double weight : w) {
                weight = generateInitialWeight();
                System.out.println(weight);
            }

    }

    private double generateInitialWeight()//method to generate small initial weights
    {
        Random rand = new Random();

        int numerator = rand.nextInt(10);

        double initialWeight = numerator/1000.0;

        if(initialWeight == 0)
            initialWeight += 0.05;

        return initialWeight;
    }

    private double sigmoid(double x)
    {
        return (1.0/1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x)
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }

}
