### Templates
#### DP
```
dp = ... # create dp array, add padding if needed
dp[0][0] = ... # init dp array and base cases

for i ...
  for j ...
    ...
    dp[i][j] = ... # transition
return dp[[n][m]
```

#### Recursion with memorization
```
mem = ... # create mem dict

def dp(i, j, ...):
  if base_case(i, j): return ... # base case
  if (i, j) not in mem:
    mem[(i, j)] = ... # transition
  return mem[(i, j)]

return dp(n, m)
```

### Types
#### 1D, 2 sets of sub-problems
可以是从左往右扫描或者右往左扫描，最后合并

#### 1D, with multiple states dp[i][0] dp[i][1],..., i is the problem size


