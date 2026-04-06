# GPT/Claude generated; context, prompt Erin Spencer
# Date: 2026-04-06

import math

def is_prime(n):
    """Return True if n is a prime number, else False."""
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def prime_numbers_up_to(limit):
    """Return a list of all prime numbers up to a given limit."""
    return [n for n in range(2, limit + 1) if is_prime(n)];