### Red Black Tree
A red black tree is a special type of balanced tree. Maximum height is O(logn) It contains following features
* A node is either red or black
* The root and leaves (nil) are black
* If a node is red, then its children are black
* All paths from a node to its nil descendants have same number of black nodes

Extra notes
* Nodes require one storage bit to keep track of its color
* The longest path (root to farthest nil) is no more than twice the length of the shortest path (root to nearest nil)

Time complexity <br>
Search, insert and remove are all O(logn)

#### Rotation
Left and right rotation. Time complexity is O(1) because we only change a few pointers. After rotation the binary search tree characteristic remains: right child's value is larder than the parent and left child's value is smaller than parent

#### Insertion
* Insert Z and color it red
* Recolor and rotate node to fix violation
* 
Scenarios
* Z is the root: color it black
* Z's uncle is red: recolor Z's uncle, parent and grandparent
* Z's uncle is black (triangle): rotate Z's parent
* Z's uncle is black (line): rotate Z's grandparent in the opposite direction of Z and recolor parent and grandparent

Time Complexity
* insertion: O(logn)
* recolor: O(1)
* fix violation: O(logn)

Code: https://www.programiz.com/dsa/red-black-tree
