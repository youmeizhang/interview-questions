## Disjoint-set / Union-find forest

### Two methods
* Find(x): find the root/cluster-id of x
* Union(x,y): merge two clusters

### Two arrays to store information
* parents = [0,1,2,3...n]
* ranks = [0] * (n+1)

Check wheather two elements are in the same set or not in O(1). Without optimization, it would be O(n)

### Optimization
* Path compression: make tree flat
  * 当寻找某一个元素的时候，顺便把它路径上要经过的element的parent也设置为cluster的root以达到tree flat的效果。但是注意并不是所有的element存的parent都一定是最终的root，所以需要最后在call find去确认(x)
* Union by rank: merge low rank tree to high rank one
  * Rank means how many branches that a root has
  * If two sub-tree has the same rank, break tie arbitarily and increase the merged tree's rank by 1. This can reduce path compressioon overhead

### Examples
* 684 547 Leetcode
