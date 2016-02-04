# Metadiff

Metadiff is a deep learning framework designed for efficiency, flexibility and
easier development and research. The framework is very closely related to Theano and Tensorflow
in terms of both syntax and Computational Graph representation.

## Motivation
The main goal of the framework is to redesign Theano, such that it can support multiple GPUs as well
as to be able to deploy it to a cluster. Additionally, we want to move the graph optimization ot C++
rather than Python for performance gains on compilation times as well as to fully separate the
Graph manipulations from the source code generation, which to allow much more rapid development.


The main distinctive features of Metadiff are:

1. Each computation node contains information about the device it will be executed on.

2. The graph optimizer provides hints for the source code generation, such as inlining and register allocation

3. Source code generation is done via a separate Backend class which traverses the optimized graph. This allows
very easily to implement different backends and reuse the whole graph framework.

**4.** A main improvement for development perspective is including full size information on graph nodes. This is achieved by modelling
the sizes of variables by a Polynomials over the integers. The consequences of this is that any size mistake done will be
detected at compile time(of the graph) rather than run time, significantly reducing development time.

**5.** Because of point 4, all errors related to any operation can be thrown at compile time(of the graph). This means that
we can design a lot more meaningful errors to be thrown and inform the practitioner for what was the mistake, since it happens
before any optimization has occurred.

6. Currently the backend used is based on [Arrayfire](www.arrayfire.com). This means that that the framework can be deployed seamlessly
to both CUDA and OpenCL devices without any extra effort.

7. Similar to Tensorflow, each node is assigned to a Group, which main goal is to facilitate a much better visualization of the graph.


## Current state of development
The framework is still in early development and there are number of things that need to be implemented:

* [ ] Indexing and slicing operators
* [ ] Full test suite
* [ ] Convolution and max pooling operators
* [ ] Other special Neural Networks operators
* [ ] A symbolic for loop (SCAN)
* [ ] Groups implementation on the nodes for better visualization
* [ ] Reuse some better visualization html library
* [ ] Implement all of the operators in the `ArrayfireBackend`
* [ ] Benchmark on several standard problems
* [ ] Add Travis integration
* [ ] Proper documentation

## Long term goals
* [ ] Multi GPU support
* [ ] Automatic *palatalization* over multiple devices
* [ ] More graph optimizations
* [ ] Real time visualization during training
* [ ] Implementing a session, which to allow for snapshots and restores of previous states

## Examples
Need to improve on this. Currently there is a single example in the `examples` folder which is the standard
Hinton autoencoder applied to MNIST.

## License
The framework is licensed under the MIT License.

