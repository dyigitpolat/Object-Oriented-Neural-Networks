/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetworkdemo;

import java.util.ArrayList;

/**
 *
 * @author yigitpolat
 */
public class NeuralNetwork 
{
    private final ArrayList<Neuron> inputNeurons;
    private final ArrayList<ArrayList<Neuron>> hiddenLayers;
    private final ArrayList<Neuron> outputNeurons;
    private final int numInputs;
    private final int numOutputs;
    private ArrayList<double[]> dataList;
    private ArrayList<double[]> targetList;
    
    public NeuralNetwork( double learningRate, int numInputs, int numOutputs, int... hidden)
    {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        dataList = new ArrayList<>();
        targetList = new ArrayList<>();
        
        inputNeurons = new ArrayList<>();
        for( int i = 0; i < numInputs; i++)
        {
            inputNeurons.add( new Neuron(1.0));
        }
        
        hiddenLayers = new ArrayList<>();
        for( int i = 0; i < hidden.length; i++)
        {
            hiddenLayers.add( new ArrayList<>());
            for( int j = 0; j < hidden[i]; j++)
            {
                if( i == 0)
                {
                    Neuron n = new Neuron( true, learningRate, inputNeurons.toArray( new Neuron[0]));
                    hiddenLayers.get(0).add( n);
                }
                else
                {
                    Neuron n = new Neuron( true, learningRate, hiddenLayers.get(i-1).toArray( new Neuron[0]));
                    hiddenLayers.get(i).add( n);
                }
            }
        }
        
        outputNeurons = new ArrayList<>();
        for( int i = 0; i < numOutputs; i++)
        {
            Neuron n = new Neuron( true, learningRate, hiddenLayers.get( hidden.length -1).toArray( new Neuron[0]));
            outputNeurons.add( n);
        }
    }
    
    public void addData( double[] data, double[] target)
    {
        if( data.length == numInputs && target.length == numOutputs)
        {
            this.dataList.add( data);
            this.targetList.add( target);
        }
    }
    
    public void addData( ArrayList<double[]> dataList, ArrayList<double[]> targetList)
    {
        for( int i = 0; i < dataList.size(); i++)
        {
            addData( dataList.get(i), targetList.get(i));
        }
    }
    
    public void trainNetwork( int numPasses)
    {
        for( int i = 0; i < numPasses; i++)
        {
            int r = (int) (Math.random() * Integer.MAX_VALUE) % dataList.size();
            
            double[] data = dataList.get(r);
            double[] target = targetList.get(r);
            double[] dErr = new double[ numOutputs];
            double mse = 0;
            double mse2 = 0;
            
            for( int j = 0; j < outputNeurons.size(); j++)
            {
                Neuron output = outputNeurons.get(j);
                double c = target[j];
                double y = output.output();
                mse += 0.5*(y - c)*(y - c);
                dErr[j] = (y - c); //Err = 0.5*(y - c)*(y - c)
            }
            
            for( int j = 0; j < inputNeurons.size(); j++)
            {
                inputNeurons.get(j).setInput( data[j]);
            }
            
            for( int j = 0; j < outputNeurons.size(); j++)
            {
                outputNeurons.get(j).backPropagate( dErr[j] );
            }

            for( int j = 0; j < outputNeurons.size(); j++)
            {
                double y2 = outputNeurons.get(j).output();
                double c = target[j];
                mse2 += 0.5*(y2 - c)*(y2 - c);
            }   
            
            for( int j = 0; j < outputNeurons.size(); j++)
            {
                double m = (mse - mse2);
                outputNeurons.get(j).setMomentum( outputNeurons.get(j).getMomentum()*0.9 + m);
            }
        }
    }
    
    public double[] classify( double[] data)
    {
        double[] scores = new double[numOutputs];
        
        for( int i = 0; i < inputNeurons.size(); i++)
        {
            inputNeurons.get(i).setInput( data[i]);
        }
        
        for( int i = 0; i < outputNeurons.size(); i++)
        {
            scores[i] = outputNeurons.get(i).output();
        }
        
        return scores;
    }
    
    
}
