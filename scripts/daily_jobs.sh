#!/usr/bin/env bash

kaggle kernels pull code1110/numeraisignals-add-features-to-targets -m -p test
kaggle kernels push -p test
rm -r test