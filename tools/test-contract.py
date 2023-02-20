import os
import json
import subprocess


# everywhere in this file 'contract' is a pure name without .sol or .wasm


# runs wasm contract in wasmtime with string input and returns dict: {"status" : "<status>", "output": "<output>"}
def run_in_wasmtime(contract: str, inpt: str) -> json:
    name = inpt[:inpt.find(' ')]
    args = inpt[inpt.find(' ') + 4:-1]
    args = args[1:-1]
    res = subprocess.run(["wasmtime", "--allow-unknown-exports", f"{contract}.wasm", "--invoke", name, "--", args, "/dev/null"], capture_output=True)
    output = bytes.fromhex(res.stderr.decode().split('\n')[-2][9:]).decode()
    json_out = json.loads(output)
    return json_out


# runs local Ethereum network and deploys contract to it
def run_in_geth(contract: str, inpt: str) -> json:
    pass


# returns name of compiled contract
def compile_to_wasm(contract: str) -> str:
    os.system(f"./evm2near {contract}.sol -o {contract}.wasm -b wasi")
    return contract + ".wasm"


def compile_to_geth(contract: str):
    pass


def extract_inputs(contract: str):
    with open(f'test_inputs/{contract}.txt', 'r') as inputsfile:
        inputs = inputsfile.readlines()
        return list(map(lambda inpt: {'name': inpt[:inpt.find(' ')], 'args': inpt[inpt.find(' ') + 5:-2]}, inputs))


def test_contract(contract: str):
    pass




if __name__ == "__main__":
    # # maybe it make sencse to paralelize it ?
    # for contract in os.listdir('test'):
    #     test_contract(contract)

    print(extract_inputs('calc'), sep='\n')
