# Usability Testing Report

## Test Participants

### Participant 1

* Name: Friend/Classmate 1
* Testing Duration: 10 Minutes

### Participant 2

* Name: Friend/Classmate 2
* Testing Duration: 10 Minutes

---

## Observations

### Participant 1 Confusion Points

1. The user was not sure what values to enter for Glucose, BMI, and Insulin.
2. The user did not understand the expected input format.
3. The user wanted example inputs before entering patient details.

### Participant 2 Confusion Points

1. The user did not know why the prediction result was "Diabetic" or "Non-Diabetic".
2. The user expected more descriptive error messages.
3. The user was unsure where the dashboard charts were saved.

---

## Bugs Found

1. Invalid text input caused the program to show an error message.
2. Extremely high or unrealistic values produced predictions but may not be medically meaningful.
3. The application does not currently restrict unrealistic input ranges.

---

## Planned Fixes

1. Add sample input examples for users.
2. Add acceptable value ranges for each field.
3. Improve error messages with clearer instructions.
4. Display the location where dashboard charts are saved.
5. Add explanations for prediction results.

---

## Conclusion

The project is functional, but usability can be improved by providing clearer instructions, better validation, and more informative output messages.
