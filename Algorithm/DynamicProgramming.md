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

#### 1. Input O(n)
dp[i] only depends on const smaller problems <br>
Time complexity: O(n) <br>
Space complexity: O(n) -> O(1)
```
dp = [0] * n
for i in range(n):
  dp[i] = f(dp[i-1], dp[i-2], ...)
return dp[n]
```

#### 2. Input O(n)
dp[i] depends on all smaller problems (dp[0], dp[1], ..., dp[i-1]) <br>
Time complexity: O(n^2) <br>
Space complexity: O(n)
```
Template

dp = new int[n]
for i in range(n):
  for j in range(i):
    dp[i] = max/min(dp[i], f(dp[j]))
return dp[n]
```

#### 3. Input O(m) + O(n), two arrays/strings
dp[i][j] depends on const smaller problems <br>
Time complexity: O(mn) <br>
Space complexity: O(mn)
```
Template

dp = new int[n+1][m+1] # init DP array
for i in range(n+1):
  for j in range(m+1):
    dp[i+1][j+1] = f(dp[i][j], dp[i][j+1], dp[i+1][j])
return dp[n][m]
```

#### 4. O(n)
Solution of a subproblem (A[i->j]), a sub-array of the input
找k作为分割点


