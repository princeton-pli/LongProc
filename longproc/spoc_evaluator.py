import json
import hashlib
import subprocess
import os

from os.path import join
from tqdm import tqdm
from subprocess import Popen, PIPE

_IMPORT_HEADER = """
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <fstream>
#include <climits>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <bitset>
using namespace std;
""".lstrip()


def hash_of_code(code, size=16):
    val = hashlib.sha1(code.encode("utf-8")).hexdigest()
    return val[-size:]

def get_executable_name(code):
    full_program = _IMPORT_HEADER + code
    file_name = hash_of_code(full_program)
    return join("tmp", file_name + ".bin")

def compilation_sanity_check(code):
    # program 
    full_program = _IMPORT_HEADER + code
    # full_program = header + ex.code_lines

    file_name = hash_of_code(full_program)
    cpp_file = join("tmp", file_name + ".cpp")
    exe_file = join("tmp", file_name + ".bin")
    with open(cpp_file, 'w') as f:
        f.write(full_program)
    # compile
    try:
        compile_out = subprocess.check_output(['g++', '-std=c++11', '-O', '-o', exe_file, cpp_file], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(" ".join(['g++', '-std=c++11', '-O', '-o', exe_file, cpp_file]))
        raise e
    return exe_file

def execution_sanity_check(exe_file, testcases, max_check=10, clean=False):
     
    for c in testcases[:max_check]:
        proc = Popen([exe_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        inputs, outputs = c
        msg = ('\n'.join(inputs) + '\n').encode('utf-8')
        try:
            proc_out, proc_err = proc.communicate(msg, timeout=3)
            proc_out = proc_out.decode('utf-8')
            proc_err = proc_err.decode('utf-8')
            expected_out = '\n'.join(outputs) + '\n'

            if proc_out != expected_out:
                return False, 'inconsistent'
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, 'timeout'
        except subprocess.CalledProcessError:
            return False, 'exec error'
        except UnicodeDecodeError:
            return False, 'decode error'
        except:
            return False, 'unknown error'

    if clean:
        os.remove(exe_file)
    return True, ''

def evaluate_spoc_code(code, testcases):
    full_program = _IMPORT_HEADER + code
    # full_program = header + ex.code_lines

    file_name = hash_of_code(full_program)
    if not os.path.exists("spoctmp"):
        os.makedirs("spoctmp")
    cpp_file = join("spoctmp", file_name + ".cpp")
    exe_file = join("spoctmp", file_name + ".bin")
    with open(cpp_file, 'w') as f:
        f.write(full_program)
    # compile
    try:
        compile_out = subprocess.check_output(['g++', '-std=c++11', '-O', '-o', exe_file, cpp_file], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return False, 'compilation error'

    status, error_msg = execution_sanity_check(exe_file, testcases, clean=True)
    return status, error_msg

