#!/usr/bin/env bash
kaggle kernels pull code1110/datacrunch-neural-network-starter-sub -m -p test
kaggle kernels push -p test
rm -r test