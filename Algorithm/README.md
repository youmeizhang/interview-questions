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
### Lame's Theorem
If Euclid's Algorithm requires `k` steps to compute the GCD of some pairs, then the smaller number in the pair must be greater than or equal to the `kth` Fibonacci number

### Difference between modulo and remainder
```
c = a/b
r = a - c*b
```
两者不同之处在于，求余运算在取`c`的值时，向0方向舍入，而取模运算在计算`c`的值时，向负无穷大方向舍入。c语言中%表示求余运算而python中则取模。如果`a`和`b`两者的符号一致，那么求余和求模结果时一致的。
```
c = -3/4 = -1
r = -3 - (-1*4) = 1
```

### Fermat's Little Theorem
If `n` is a prime number and `a` is any positive integer less than `n`, then `a` raised to the `nth` power is congruent to `a` modulo `n`

The answer from Fermat test is only probably correct. But it is appled to the field of cryptography such as `RSA Algorithm`











