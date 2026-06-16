"""
Basic Tests for Healthcare Diabetes Project
"""

from diabetes_prediction import validate_input

print("===== TEST 1: Input Validation =====")

try:
    validate_input(120, "Glucose")
    print("PASS")
except:
    print("FAIL")


print("\n===== TEST 2: Empty Input =====")

try:
    validate_input(None, "Glucose")
    print("FAIL")
except:
    print("PASS")


print("\n===== TEST 3: Wrong Data Type =====")

try:
    validate_input("abc", "Glucose")
    print("FAIL")
except:
    print("PASS")


print("\n===== ALL TESTS COMPLETED =====")