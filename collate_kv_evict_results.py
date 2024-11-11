import os
import glob
import json

import click

@click.command()
@click.option("--results-root", default="/home/users/akane/kv-evict-results", type=str, help="Root directory of all results")
@click.option("--output-filepath", default="/home/users/akane/kv-evict-results-collated.json", type=str, help="Root directory of all results")
def main(results_root, output_filepath):
    """Collates all KV eviction runs' results in one json file"""
    results_dict = dict()
    all_dirs = glob.glob(os.path.join(results_root, "*"))
    
    for dirpath in all_dirs:
        if dirpath.startswith("vit"):
            print(dirpath)
    

if __name__ == "__main__":
    main()
