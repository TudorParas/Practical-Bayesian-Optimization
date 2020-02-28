language: PYTHON
name:    "svm"

variable {
 name: "C"
 type: FLOAT
 size: 1
 min:  0.1
 max:  10000.0
}

variable {
 name: "EPSILON"
 type: FLOAT
 size: 1
 min:  0.001
 max:  0.1
}


