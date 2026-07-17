from planner import plan
from coder import code

problem = "Write a function that returns True if a string is a palindrome."
p = plan(problem)
print("PLAN:\n", p)
c = code(problem, p)
print("CODE:\n", c)