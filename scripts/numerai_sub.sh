#!/usr/bin/env bash

# ---------------------------
# Numerai Tournament
# ---------------------------
# get tournament data
# kaggle kernels pull code1110/numerai-test-to-feather-nomi -m -p test
kaggle kernels pull code1110/numerai-fetch-data-and-fe -m -p test
kaggle kernels push -p test

sleep 30m

rm -r test

# submissions
kaggle kernels pull code1110/numerai-base-subs -m -p test
kaggle kernels push -p test
rm -r test

sleep 3m

kaggle kernels pull code1110/numerai-base-nn-subs -m -p test
kaggle kernels push -p test
rm -r test

sleep 3m

# kaggle kernels pull code1110/numerai-gbdt-and-subs -m -p test
# kaggle kernels pull code1110/numerai-gbdt-and-subs-r259 -m -p test
kaggle kernels pull code1110/numerai-gbdt-subs-r261 -m -p test
kaggle kernels push -p test
rm -r test

sleep 3m

# # kaggle kernels pull code1110/numerai-nn-and-subs -m -p test
# kaggle kernels pull code1110/numerai-nn-and-subs-r259 -m -p test
kaggle kernels pull code1110/numerai-nn-subs-r261 -m -p test
kaggle kernels push -p test
rm -r test

sleep 40m

# # kaggle kernels pull code1110/numerai-emsemble-and-subs -m -p test
# kaggle kernels pull code1110/numerai-emsemble-and-subs-r259 -m -p test
kaggle kernels pull code1110/numerai-emsemble-subs-r261 -m -p test
kaggle kernels push -p test
rm -r test

# ---------------------------
# Numerai Signals Submission
# ---------------------------
kaggle kernels pull code1110/numerai-signals-stats -m -p test
kaggle kernels push -p test
rm -r test

kaggle kernels pull code1110/numerai-signals-prophet -m -p test
kaggle kernels push -p test
rm -r test

kaggle kernels pull code1110/numeraisignals-add-features-to-targets -m -p test
kaggle kernels push -p test
rm -r test

sleep 90m

# ---------------------------
# DataCrunch Submission
# ---------------------------
kaggle kernels pull code1110/datacrunch-ensemble-sub20210411 -m -p test
kaggle kernels push -p test
rm -r test
