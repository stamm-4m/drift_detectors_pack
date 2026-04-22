import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.join(".."))
sys.path.insert(0, os.path.join("..", ".."))
sys.path.insert(0, os.path.join("..", "..", ".."))

from drift_detectors.univariate.psi.detector import PSI

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_dir,"indepensim_data")
reference_dir = os.path.join(data_dir,"reference_data")
test_dir = os.path.join(data_dir,"test_data")


'''
The idea behind this is we provide use cases where we use drift detectors with real data.
It proves they work, and shows how they can be practially used.
See: https://gitlab.com/stamm-4m/drift_detectors_pack/-/issues/10
'''
allowed_columns = ["penicillin_concentration"]

def _load_csv(fn):
    with open(fn, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
        headers = reader.fieldnames
        return headers, data
    

def _numeric_columns(headers, data):
    cols = {}
    for col in headers:
        vals = []
        if col == "experiment_ID":
            continue
        for row in data:
            cell = row.get(col, '').strip()
            if cell == '':
                continue
            try:
                vals.append(float(cell))
            except ValueError:
                vals = None
                break
        if vals:
            cols[col] = np.array(vals)
    return cols

results = []

num_tests = 0
num_datasets = 0
num_drift_found = 0
detector = PSI()

results = {}
for ref_fn in os.listdir(reference_dir):
    ref_path = os.path.join(reference_dir, ref_fn)
    headers_ref, data_ref = _load_csv(ref_path)
    num_ref = _numeric_columns(headers_ref, data_ref)
    for test_fn in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_fn)
        headers_test, data_test = _load_csv(test_path)
        num_test = _numeric_columns(headers_test, data_test)
        num_datasets += 1
        for col in num_ref:
            if col not in allowed_columns:
                continue
            if col not in num_test:
                continue
            ref_arr  = num_ref[col]
            test_arr = num_test[col]
            num_tests += 1
            res = detector.calculate(test_arr, ref_arr)
            if res.drift:
                if ref_fn not in results:
                    results[ref_fn] = {}
                if test_fn not in results[ref_fn]:
                    results[ref_fn][test_fn] = []
                results[ref_fn][test_fn].append((col, res))
                num_drift_found += 1




for ref,tsts in results.items():
    print(f'\n\n--- Reference Data: {ref} ---')
    for tst,params in tsts.items():
        print(f'\n- Test Data: {tst} -')
        for param,data in params:
            print(param,data.score,data.details)
print("\n\n")
print(f"Finished testing with {num_datasets} dataset comparisons and {num_tests} drift detections performed.")
print(f"Found {num_drift_found} drifts {(num_drift_found/num_tests)*100}% ")