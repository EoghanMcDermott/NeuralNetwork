public class Main {

    public static void main(String[] args)
    {
       MLP nn = new MLP(2,2,1);

       double[][] input =  {{0,0}, {0,1}, {1,0}, {1,1}};
       double[] output = {0, 1, 1, 0};

       int maxEpochs = 1500;

        for (int e=0; e<maxEpochs; e++)
        {
            double error = 0;

            for (int i = 0; i < input.length; i++)
            {
                nn.forwardPass(input[i]);
                error += nn.backProp(output[i]);
                System.out.println("Expected: " + output[i]);
            }

            nn.updateWeights(0.00075);

            System.out.println("Error at epoch " + e + ": " + error + "\n");
        }
    }
}
