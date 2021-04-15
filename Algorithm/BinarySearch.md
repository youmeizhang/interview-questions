Template [l, r)

```
def binary_search(l, r):
  while l < r:
    if f(m): return m # optional
    if g(m):
      r = m # [l, m)
    else:
      l = m + 1 # [m+1, r)
  return l # or not found

binary_search(0, len(A))
```

Example: return the lower_bound / upper_bound of a val in a sorted array
* lower_bound(x): first index of i, such that A[i] >= x
* upper_bound(x): first index of i, such that A[i] > x  <br>
A = [1, 2, 2, 2, 4, 4, 5] <br>
lower_bound(A, 2) = 1, lower_bound(A, 3) = 4 (does not exist) <br>
upper_bound(A, 2) = 4, upper_bound(A, 5) = 7 (does not exist)
```
def lower_bound(A, val, l, r):
  while l < r:
    m = l + (r - l) // 2
    if A[m] >= val: # g(m)
      r = m
    else:
      l = m + 1
  return l

def upper_bound(A, val, l, r):
  while l < r:
    m = l + (r - l) // 2
    if A[m] > val: # g(m)
      r = m
    else:
      l = m + 1
  return l
```
