import argparse
dsize = 512

def out():
    print(dsize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    args = parser.parse_args()
    
    print(dsize)
    dsize = args.size
    
    print(dsize)
    
    out()