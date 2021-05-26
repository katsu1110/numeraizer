#!/usr/bin/env bash

kaggle kernels pull code1110/api-daily-jobs -m -p test
kaggle kernels push -p test
rm -r test