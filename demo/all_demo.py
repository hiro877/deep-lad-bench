import os
import glob

if __name__ == '__main__':
    benchmarks = glob.glob("../benchmarks/*/*.sh")
    skip_words = "w_semantics htm spclf".split()
    dataset = "BGL"

    for bench in benchmarks:
        for skip_word in skip_words:
            if skip_word in bench:
                break
        else:
            if dataset:
                with open(bench, mode='r') as f:
                    for line in f.readlines():
                        if dataset in line:
                            print(line)
                            os.system(f'{line}')
            else:
                print(f'bash {bench}')
                os.system(f'bash {bench}')
