#fibonaccy using dynamic programming. The major difference is that you save the values in a cache. 
def fibonacci_dp(n):
    # Table for the already calculated values
    fib = [0] * (n + 1)
    fib[0] = 0
    if n > 0:
        fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    print("I'm here")
    return fib[n]

print("Fibonacci(10) =", fibonacci_dp(10))