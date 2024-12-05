import os
import glob
import shutil

def main():
    INITIAL_RESULTS_PATH = "/home/users/akane/kv-evict-3spot-policy"
    FINAL_RESULTS_PATH = "/home/users/akane/kv-evict-results-3spot-policy"
    for dirname in os.listdir(INITIAL_RESULTS_PATH):
        frompath = os.path.join(INITIAL_RESULTS_PATH, dirname, "unpruned")
        topath = os.path.join(FINAL_RESULTS_PATH, dirname, "unpruned")
        
        if os.path.exists(os.path.join(FINAL_RESULTS_PATH, dirname)):
            shutil.copytree(frompath, topath)


if __name__ == "__main__":
    main()
