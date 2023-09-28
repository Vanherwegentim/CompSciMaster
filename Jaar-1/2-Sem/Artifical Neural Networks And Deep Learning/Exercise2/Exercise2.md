## Exercise session 2 Hopfield networks

**Create a Hopfield network with attractors T = [1 1; −1 − 1; 1 − 1]T and the corresponding number of neurons. You can use script rep2 as a basis and modify it to start from some particular points (e.g. of high symmetry) or to generate other numbers of points. Start with various initial vectors and note down the obtained attractors after a sufficient number of iterations. Are the real attractors the same as those used to create the network? If not, why do we get these unwanted attractors? How many iterations does it typically take to reach the attractor? What can you say about the stability of the attractors?**

When we enter those as attractors, another spurious state is added. This state will be [-1 1]. This unwanted attractor is a consequence of the way hopfield networks work.



**Do the same for a three neuron Hopfield network. This time use script rep3.**





**The function hopdigit creates a Hopfield network which has as attractors the handwritten digits 0, · · · , 9. Then to test the ability of the network to correctly retrieve these patterns some noisy digits are given to the network. Is the Hopfield model always able to reconstruct the noisy digits? If not why? What is the influence of the noise on the number of iterations?**

The hopfield model is not always able to retrieve the correct patterns. If the noise is higher it will require more iterations but when the noise gets high enough even that isn't a guarantee anymore.