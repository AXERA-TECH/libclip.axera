import os
import json
import glob
import base64
import argparse

from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path')
    parser.add_argument('--text', type=str, help='')
    args = parser.parse_args()


    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    except:
        tokenizer = None
    if None == tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        except:
            tokenizer = None
    if None == tokenizer:
        print("Default load tokenizer failed for ", args.tokenizer_path)
    
    if None != tokenizer:
        ids = tokenizer.encode(args.text)
        print("ids size: ", len(ids))
        print("ids: ", ids)
        
        text = tokenizer.decode(ids)
        print("text: ", text)
        
        # text = tokenizer.decode([0, 994, 1322, 2])
        # print("text: ", text)

if __name__ == '__main__':
    main()