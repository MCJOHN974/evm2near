### This script assumes that evm2near is already compiled

import os


print(os.listdir("tools/benchmark/inputs"))

contracts = list(map(lambda x: x[:-5], os.listdir("tools/benchmark/inputs")))

print(f"contracts = {contracts}")


def compile(name: str):
    os.system(f'./evm2near test/{name}.sol -o {name}.wasm -b near')


def copy(name: str):
    os.system(f'cp {name}.wasm tools/benchmark/{name}.wasm')


def remove(name: str):
    os.system(f'rm tools/benchmark/{name}.wasm')


def compile_contracts():
    for contract in contracts:
        compile(contract)


def copy_contracts():
    for contract in contracts:
        copy(contract)


def clean():
    for contract in contracts:
        remove(contract)


def run_bench():
    os.chdir('tools/benchmark')
    os.system('cargo run')
    os.chdir('../../')




import pandas as pd


if __name__ == "__main__":
    clean()
    compile_contracts()
    print("Contracts compiled")
    copy_contracts()
    print("Benchmark started")
    run_bench()
    print("Benchmark ended, see results in tools/benchmark/pages/index.html")
    print("Clean started")
    clean()
    print("Clean ended")
    print(f"current dir = {os.getcwd()}")



    tables = []
    for file in os.listdir("tools/benchmark/csvs"):
        if file[-4:] != ".csv":
            continue
        print(f"current dir = {os.getcwd()}")
        tables.append(pd.read_csv(f"tools/benchmark/csvs/{file}"))

    table = pd.concat(tables)
    table.to_html("tools/benchmark/pages/index.html")
    html_file = table.to_html()
