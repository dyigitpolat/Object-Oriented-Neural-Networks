/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetworkdemo;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author yigitpolat
 */
public class Neuron {
    
    private ArrayList<Neuron> inputSynapses;
    private double[] weights;
    private final boolean isInputNeuron;
    private double input;
    private double learningRate;
    private double momentum;
    
    public Neuron( double input)
    {
        isInputNeuron = true;
        this.input = input;
        weights = new double[1];
        weights[0] = 1;
        learningRate = 0;
        momentum = 0;
    }
    
    public Neuron( boolean hasBias, double learningRate, Neuron... neurons)
    {
        isInputNeuron = false;
        inputSynapses = new ArrayList<>();
        inputSynapses.addAll( Arrays.asList(neurons));
        if( hasBias)
        {
            inputSynapses.add( new Neuron( 1.0));
        }
        
        weights = new double[ inputSynapses.size()];
        this.learningRate = learningRate;
        
        for( int i = 0; i < weights.length; i++)
        {
            weights[i] = Math.random() * 1.0 - 0.5;
        }
        
        momentum = 0;
    }
    
    public double getMomentum()
    {
        return momentum;
    }
    
    public void setMomentum( double m)
    {
        if( isInputNeuron)
            return;
        
        momentum = m;
        for( Neuron n : inputSynapses)
        {
            n.setMomentum( m);
        }
    }
    
    public void setInput( double input)
    {
        if( isInputNeuron)
            this.input = input;
    }
    
    public void backPropagate( double dErr)
    {
        if( isInputNeuron) 
        {
            return; 
        }
        
        double hj = inputSum();
        double yj = output();
        for( int i = 0; i < inputSynapses.size(); i++)
        {
            double xi = inputSynapses.get(i).output();
            double dErrj = activationDerivative( hj)*dErr;
            double deltaW = (learningRate + momentum)*dErrj*xi;
            
            inputSynapses.get(i).backPropagate( dErrj*weights[i] ); //beware, recursion.
            weights[i] -= deltaW;
        }
    }
    
    public double inputSum()
    {
        double dotProduct = 0;
        int size = inputSynapses.size();
        for( int i = 0; i < size; i++)
        {
            dotProduct += inputSynapses.get(i).output() * weights[i];
        }
        
        return dotProduct;
    }
    
    public double output()
    {
        if( isInputNeuron)
            return input;
        
        return activation( inputSum());
    }
    
    public double[] getWeights()
    {
        return weights;
    }
    
    public void setWeights( double... w)
    {
        weights = w;
    }
    
    public void printWeightsRecursively( int i, int j)
    {
        if( isInputNeuron)
        {
            return;
        }
        
        System.out.println( "LAYER: " + i + ", NEURON: " + j);
        for( double w : weights)
        {
            System.out.println( w);
        }
        
        if( j == 0)
            for( Neuron n : inputSynapses)
            {
                n.printWeightsRecursively( i + 1, j++);
            }
    }
    
    private double activationDerivative( double x) //sigmoid
    {
        return activation(x)*(1.0 - activation(x));
    }
    
    private double activation( double arg) //sigmoid
    {
        return 1.0/(1 + Math.exp( -arg));
    }
    
}
