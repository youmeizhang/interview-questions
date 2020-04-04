## Algorithm

### Euclid's Algorithm
Observation: if `r` is the remainder when `a` is divided by `b`, then the common divisors of `a` and `b` are precisely the same as the common divisor of `b` and `r`. `Gcd(a, b) = Gcd(b, r)`

Alternative expression: 两个整数的最大公约数等于其中较小的数和两数的差的最大公约数 
```Python
# Recursive
def Gcd (a, b):
  if b == 0:
    return a
  return Gcd(b, a % b)
  
# iterative
def Gcd(a, b):
  while b != 0:
    r = b
    b = a % b
    a = r
  return a
```
