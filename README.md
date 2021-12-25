# DirectedWeigthedGraph_2 - this time in python! :)

This is a repository of DirectedWeightedGraph as part of an assignment in Ariel University 
</br>

* [Shaked's github](https://github.com/20shaked20)
* [Yonatan's github](https://github.com/Teklar223)

## Introduction
- If you just want to know how to use this code please skip to the ``` How To Use ``` segment below.
- If you want to know how to work with the gui, we've dedictated a wiki page for it.

In this assignment we are expected to construct and implement solutions to known problems in the subject of Graphs, specifically Directed and Weighted graphs (see links at the end of the readme), and also represent the graph with a gui of our own making, in this exercise we were given free reign on things like which libraries to use and the worst case runtime complexity of algorithms (naturally we are expected to do our best and not concoct O(n!) solutions).

## Approach

This assignment is essentially a "copy" of [this assignment](https://github.com/20shaked20/DirectedWeightedGraph) but in python, the idea is for us to familiarize ourselves with the workings of both java and python and as such we didnt have to think too hard on the algorithms, but rather on the implementation differences in python vs in java.</br>

Despite this we felt the need to improve on our TSP solution and experimented with different algorithms, one such attempted solution was with Simulated Annealing - a basic machine learning algorithm, unfortunetly this attempt was flawed, and due to to time restrictions we decided to use a more naive and simple approach, a greedy algorithm that guarantee's (heuristically) a feasible solution.
 </br>

## The Algorithms
We will not go into the heavy depths of each algorithm, but rather describe what solution we used. </br>

- ``` shortestPath(int src, int dest) ``` and ``` shortestPathDist(int src, int dest) ``` : </br>
Using the dijkstra algorithm we managed to get accurate results in ``` O(|V| * (|E| + |V|)) ```, using the pseudo-code located in the wiki page as a guideline for implementation and private properties of nodes such as tag, info, and weight.
- ``` center() ``` : </br>
Using the dijkstra algorithm to calculate the sum of distances from each node the every other one, and then dividing by the amount of nodes to get an average. </br>
after that we simply iterate to find the node with the best average distance to every other node.
- ``` tsp(List<NodeData> cities) ``` : </br>
Firstly, TSP is an np-hard problem and thus difficult to find an accurate solution for, and in this exercise we were allowed to visit a city (node) more than once </br>
Therefore we used a heuristic and greedy algorithm that *guarantees a solution* - for each node it reaches it chooses the lowest weight neighbour and if it detects a loop or gets stuck it calculates the shortest path from the current node to one of the nodes that were not visited yet using Dijkstra's algorithm, this approach tends to repeat some steps but this is traded for a guaranteed and feasible solution.  </br>

## The Classes
For nearly every class there is an interface object in the api folder, which goes into detail about the methods the classes must have and what they do, so unless it's not stated somewhere we will not elaborate

- ``` DiGraph implements GraphInterface```
- ``` GraphAlgo implements GraphAlgoInterface```
- ``` Node ``` 
- ``` main ```: </br>
this class is used to by the course staff to check our implementation using the ``` check() ``` method ( which calls ``` check0() ... check3() ``` )

## How To use

### If you're using PyCharm and scientific mode is on:
it may interfere with the gui, if this happens please follow these [instructions](https://stackoverflow.com/questions/48384041/pycharm-how-to-remove-sciview)

## Lessons Learned
### things to improve
- Better groundwork preparation
- If we choose to use TDD than prepare tests before code 
- More accurate tests

### things to keep
- Good source control
- Proper divide of work
- Keeping track of exercise additions and changes

## File Hierarchy
![image](https://user-images.githubusercontent.com/73063105/147392899-a9b84ab5-a327-4353-807f-ac2cf9e38d74.png)


## Reading Material
- About Directed, Weighted, and Directed + Weighted graphs: http://math.oxford.emory.edu/site/cs171/directedAndEdgeWeightedGraphs/
- Shortest Path: https://en.wikipedia.org/wiki/Shortest_path_problem#Algorithms
- Dijkstra: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- Graph Center: https://en.wikipedia.org/wiki/Graph_center
- Travelling Salesman Problem (TSP): https://en.wikipedia.org/wiki/Travelling_salesman_problem
- Gson library: https://github.com/google/gson
- Heuristics https://en.wikipedia.org/wiki/Heuristic_(computer_science)
