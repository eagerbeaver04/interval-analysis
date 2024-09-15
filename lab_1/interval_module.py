import numpy as np


class Interval:
    def __init__(self, lower, upper):
        if lower > upper:
            raise ValueError("Lower bound cannot be greater than upper bound.")
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f"[{self.lower}, {self.upper}]"

    # Interval midpoint
    def mid(self):
        return (self.lower + self.upper) / 2

    # Interval width
    def width(self):
        return self.upper - self.lower

    # Interval addition
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        return Interval(self.lower + other, self.upper + other)

    # Subtracting intervals
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        return Interval(self.lower - other, self.upper - other)

    # Interval multiplication
    def __mul__(self, other):
        if isinstance(other, Interval):
            products = np.array([
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper
            ])
            return Interval(np.min(products), np.max(products))
        return Interval(self.lower * other, self.upper * other)

    # Interval division
    def __truediv__(self, other):
        if isinstance(other, Interval):
            divisions = np.array([
                self.lower / other.lower,
                self.lower / other.upper,
                self.upper / other.lower,
                self.upper / other.upper
            ])
            return Interval(np.min(divisions), np.max(divisions))
        return Interval(self.lower / other, self.upper / other)

    # Belonging of a number to an interval
    def __contains__(self, value):
        return self.lower <= value <= self.upper

    # Intersection of intervals
    def __and__(self, other):
        new_lower = max(self.lower, other.lower)
        new_upper = min(self.upper, other.upper)
        if new_lower > new_upper:
            return Interval(0, 0)
        return Interval(new_lower, new_upper)

    # Combining intervals
    def __or__(self, other):
        return Interval(min(self.lower, other.lower), max(self.upper, other.upper))