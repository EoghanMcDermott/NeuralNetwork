public class Main {

    public static void main(String[] args)
    {
       MLP nn = new MLP(2,2,1);

       double[][] input =  {{0,0}, {0,1}, {1,0}, {1,1}};
       double[] output = {0, 1, 1, 0};

       int maxEpochs = 200;
       double learningRate = 0.1;
        for (int e=0; e<maxEpochs; e++)
        {
            double totalError = 0;

            for (int i=0;i<input.length;i++)
            {
                nn.forwardPass(input[i]);
                double thisError = nn.backProp(output[i]);
                totalError += thisError;
                System.out.println("Expected: " + output[i]);
//                System.out.println("Error: " + thisError + "\n");
            }

            nn.updateWeights(learningRate);

//            if(e%4 == 0)
                learningRate += 0.1;

            System.out.println("Total error at epoch " + e + ": " + totalError + "\n");
        }
    }
}
