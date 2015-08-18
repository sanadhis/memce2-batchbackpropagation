using System;
namespace BatchTrain{
  public class NeuralNetwork
    {
      private static Random rnd;

      private int numInput;
      private int numHidden;
      private int numOutput;

      private double[] inputs;

      private double[][] ihWeights; // input-hidden
      private double[] hBiases;
      private double[] hOutputs;

      private double[][] hoWeights; // hidden-output
      private double[] oBiases;

      private double[] outputs;

      private double[] oGrads; // output gradients for back-propagation
      private double[] hGrads; // hidden gradients for back-propagation

      // batch training accumulated deltas
      private double[][] ihAccDeltas;
      private double[] hBiasesAccDeltas;
      private double[][] hoAccDeltas;
      private double[] oBiasesAccDeltas;


      public NeuralNetwork(int numInput, int numHidden, int numOutput)
      {
        rnd = new Random(0); // for InitializeWeights()

        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;

        this.inputs = new double[numInput];

        this.ihWeights = MakeMatrix(numInput, numHidden);
        this.hBiases = new double[numHidden];
        this.hOutputs = new double[numHidden];

        this.hoWeights = MakeMatrix(numHidden, numOutput);
        this.oBiases = new double[numOutput];

        this.outputs = new double[numOutput];

        this.hGrads = new double[numHidden];
        this.oGrads = new double[numOutput];

        this.ihAccDeltas = MakeMatrix(numInput, numHidden);
        this.hBiasesAccDeltas = new double[numHidden];
        this.hoAccDeltas = MakeMatrix(numHidden, numOutput);
        this.oBiasesAccDeltas = new double[numOutput];

        this.InitializeWeights();
      } // ctor

      private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
      {
        double[][] result = new double[rows][];
        for (int r = 0; r < result.Length; ++r)
          result[r] = new double[cols];
        return result;
      }

      // ----------------------------------------------------------------------------------------

      private void SetWeights(double[] weights)
      {
        // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
        int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        if (weights.Length != numWeights)
          throw new Exception("Bad weights array length: ");

        int k = 0; // points into weights param

        for (int i = 0; i < numInput; ++i)
          for (int j = 0; j < numHidden; ++j)
            ihWeights[i][j] = weights[k++];
        for (int i = 0; i < numHidden; ++i)
          hBiases[i] = weights[k++];
        for (int i = 0; i < numHidden; ++i)
          for (int j = 0; j < numOutput; ++j)
            hoWeights[i][j] = weights[k++];
        for (int i = 0; i < numOutput; ++i)
          oBiases[i] = weights[k++];
      }

      private void InitializeWeights()
      {
        // initialize weights and biases to small random values
        int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        double[] initialWeights = new double[numWeights];
        double lo = -0.01;
        double hi = 0.01;
        for (int i = 0; i < initialWeights.Length; ++i)
          initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
        this.SetWeights(initialWeights);
      }

      public double[] GetWeights()
      {
        // returns the current set of wweights, presumably after training
        int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        double[] result = new double[numWeights];
        int k = 0;
        for (int i = 0; i < ihWeights.Length; ++i)
          for (int j = 0; j < ihWeights[0].Length; ++j)
            result[k++] = ihWeights[i][j];
        for (int i = 0; i < hBiases.Length; ++i)
          result[k++] = hBiases[i];
        for (int i = 0; i < hoWeights.Length; ++i)
          for (int j = 0; j < hoWeights[0].Length; ++j)
            result[k++] = hoWeights[i][j];
        for (int i = 0; i < oBiases.Length; ++i)
          result[k++] = oBiases[i];
        return result;
      }

      // ----------------------------------------------------------------------------------------

      private double[] ComputeOutputs(double[] xValues)
      {
        double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
        double[] oSums = new double[numOutput]; // output nodes sums

        for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
          this.inputs[i] = xValues[i];

        for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
          for (int i = 0; i < numInput; ++i)
            hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

        for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
          hSums[i] += this.hBiases[i];

        for (int i = 0; i < numHidden; ++i)   // apply activation
          this.hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

        for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
          for (int i = 0; i < numHidden; ++i)
            oSums[j] += hOutputs[i] * hoWeights[i][j];

        for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
          oSums[i] += oBiases[i];

        double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
        Array.Copy(softOut, outputs, softOut.Length);

        double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
        Array.Copy(this.outputs, retResult, retResult.Length);
        return retResult;
      } // ComputeOutputs

      private static double HyperTanFunction(double x)
      {
        if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
        else if (x > 20.0) return 1.0;
        else return Math.Tanh(x);
      }

      private static double[] Softmax(double[] oSums)
      {
        // determine max output sum
        // does all output nodes at once so scale doesn't have to be re-computed each time
        double max = oSums[0];
        for (int i = 0; i < oSums.Length; ++i)
          if (oSums[i] > max) max = oSums[i];

        // determine scaling factor -- sum of exp(each val - max)
        double scale = 0.0;
        for (int i = 0; i < oSums.Length; ++i)
          scale += Math.Exp(oSums[i] - max);

        double[] result = new double[oSums.Length];
        for (int i = 0; i < oSums.Length; ++i)
          result[i] = Math.Exp(oSums[i] - max) / scale;

        return result; // now scaled so that xi sum to 1.0
      }

      // ----------------------------------------------------------------------------------------

      private void ComputeAndAccumulateDeltas(double[] tValues, double learnRate) // for curr outputs
      {
        // 1. compute output gradients
        for (int i = 0; i < numOutput; ++i)
        {
          // derivative of softmax = (1 - y) * y (same as log-sigmoid)
          double derivative = (1 - outputs[i]) * outputs[i];
          oGrads[i] = derivative * (tValues[i] - outputs[i]); // assumes MSE
        }

        // 2. compute hidden gradients
        for (int i = 0; i < numHidden; ++i)
        {
          // derivative of tanh = (1 - y) * (1 + y)
          double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
          double sum = 0.0;
          for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
          {
            double x = oGrads[j] * hoWeights[i][j];
            sum += x;
          }
          hGrads[i] = derivative * sum;
        }

        // 3a. compute and accumulate input-hidden weight deltas
        for (int i = 0; i < numInput; ++i) 
        {
          for (int j = 0; j < numHidden; ++j) 
          {
            double delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
            this.ihAccDeltas[i][j] += delta; // accumulate
          }
        }

        // 3b. compute hidden biases deltas
        for (int i = 0; i < numHidden; ++i)
        {
          double delta = learnRate * hGrads[i] * 1.0; // 1.0 is dummy input
          this.hBiasesAccDeltas[i] += delta;
        }

        // 4a. compute hidden-output weights deltas
        for (int i = 0; i < numHidden; ++i)
        {
          for (int j = 0; j < numOutput; ++j)
          {
            double delta = learnRate * oGrads[j] * hOutputs[i];
            this.hoAccDeltas[i][j] += delta;
          }
        }

        // 4b. compute output biases deltas
        for (int i = 0; i < numOutput; ++i)
        {
          double delta = learnRate * oGrads[i] * 1.0;
          this.oBiasesAccDeltas[i] += delta;
        }

      } // ComputeAndAccumulateDeltas

      // ----------------------------------------------------------------------------------------

      public void Train(double[][] trainData, int maxEpochs, double learnRate)
      {
        // train a back-prop style NN using learning rate with batch training
        int epoch = 0;
        double[] xValues = new double[numInput]; // inputs
        double[] tValues = new double[numOutput]; // target values

        while (epoch < maxEpochs)
        {
          double mse = MeanSquaredError(trainData);
          if (mse < 0.020) break; // consider passing value in as parameter

          //if (epoch < 100)
          //  Console.WriteLine(epoch + "\t" + mse.ToString("F4"));

          // zero-out accumulated weight deltas
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihAccDeltas[i][j] = 0.0;

          for (int i = 0; i < numHidden; ++i)
            hBiasesAccDeltas[i] = 0.0;

          for (int i = 0; i < numHidden; ++i)
            for (int j = 0; j < numOutput; ++j)
              hoAccDeltas[i][j] = 0.0;

          for (int i = 0; i < numOutput; ++i)
            oBiasesAccDeltas[i] = 0.0;

          for (int i = 0; i < trainData.Length; ++i) // for each training item
          {
            Array.Copy(trainData[i], xValues, numInput); // get curr x-values
            Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get curr t-values
            ComputeOutputs(xValues); // compute outputs (store them internally)
            ComputeAndAccumulateDeltas(tValues, learnRate);
          }

          // update all weights using the accumulated deltas
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihWeights[i][j] += ihAccDeltas[i][j];

          for (int i = 0; i < numHidden; ++i)
            hBiases[i] += hBiasesAccDeltas[i];

          for (int i = 0; i < numHidden; ++i)
            for (int j = 0; j < numOutput; ++j)
              hoWeights[i][j] += hoAccDeltas[i][j];

          for (int i = 0; i < numOutput; ++i)
            oBiases[i] += oBiasesAccDeltas[i];

          ++epoch;
        }
      } // Train

      private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
      {
        // average squared error per training tuple
        double sumSquaredError = 0.0;
        double[] xValues = new double[numInput]; // first numInput values in trainData
        double[] tValues = new double[numOutput]; // last numOutput values

        // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
        for (int i = 0; i < trainData.Length; ++i)
        {
          Array.Copy(trainData[i], xValues, numInput);
          Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
          double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
          for (int j = 0; j < numOutput; ++j)
          {
            double err = tValues[j] - yValues[j];
            sumSquaredError += err * err;
          }
        }

        return sumSquaredError / trainData.Length;
      }

      // ----------------------------------------------------------------------------------------

      public double Accuracy(double[][] testData)
      {
        // percentage correct using winner-takes all
        int numCorrect = 0;
        int numWrong = 0;
        double[] xValues = new double[numInput]; // inputs
        double[] tValues = new double[numOutput]; // targets
        double[] yValues; // computed Y

        for (int i = 0; i < testData.Length; ++i)
        {
          Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
          Array.Copy(testData[i], numInput, tValues, 0, numOutput);
          yValues = this.ComputeOutputs(xValues);
          int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

          if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
            ++numCorrect;
          else
            ++numWrong;
        }
        return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
      }

      public int feedfoward(double[] testData)
      {
        double[] xValues = new double[numInput]; // inputs
        double[] tValues = new double[numOutput]; // targets
        double[] yValues; // computed Y

        Array.Copy(testData, xValues, numInput); // parse test data into x-values and t-values
        Array.Copy(testData, numInput, tValues, 0, numOutput);
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        
        return maxIndex; // ugly 2 - check for divide by zero
      }

      private static int MaxIndex(double[] vector) // helper for Accuracy()
      {
        // index of largest value
        int bigIndex = 0;
        double biggestVal = vector[0];
        for (int i = 0; i < vector.Length; ++i)
        {
          if (vector[i] > biggestVal)
          {
            biggestVal = vector[i]; bigIndex = i;
          }
        }
        return bigIndex;
      }

    } // NeuralNetwork
}