language: PYTHON
name:     "logitreg"

variable {
 name: "RATE"
 type: FLOAT
 size: 1
 min:  0.0
 max:  1.0
}

variable {
 name: "REG"
 type: FLOAT
 size: 1
 min:  0.0
 max:  1.0
}

variable {
 name: "BATCH"
 type: INT
 size: 1
 min:  20
 max:  2000
}

variable {
 name: "EPOCH"
 type: INT
 size: 1
 min:  5
 max:  2000
}

