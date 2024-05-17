import os 
import argparse
from tqdm import tqdm
from mmseg.apis import MMSegInferencer

DATA_ROOT = r'/home/gauthierli/code/temp/0402_7000/0402_clean_data'

DATA_PATH = {
    "loveda": os.path.join(DATA_ROOT, r"LoveDA/clean/img_dir/val"),
    "loveda_san": os.path.join(DATA_ROOT, r"LoveDA/clean/img_dir/val"),
    "potsdam": os.path.join(DATA_ROOT, r"Postdam/clean/img_dir/val"),
    "vaihingen": os.path.join(DATA_ROOT, r"Vaihingen/clean/img_dir/val")
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasetname', help='Name of Dataset')
    parser.add_argument('model', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('savedir', default='./show' , help='Config file')
    parser.add_argument('--spec', default=None , help='Specific img')
    args = parser.parse_args()
    
    mmseg_inferencer = MMSegInferencer(
        args.model,
        args.checkpoint,
        dataset_name=args.datasetname,
        device='cuda')
    
    img_root = DATA_PATH[args.datasetname]
    imgs_lst = os.listdir(img_root)
    if args.spec is None:
        for img in tqdm(imgs_lst):
            img_path = os.path.join(img_root, img)
            out_path = os.path.join(args.savedir, img)
            mmseg_inferencer(
                        img_path,
                        show=False,
                        out_dir=out_path,
                        opacity=1,
                        with_labels=False)
    else:
        img_path = os.path.join(img_root, args.spec)
        out_path = os.path.join(args.savedir, args.spec)
        mmseg_inferencer(
                        img_path,
                        show=False,
                        out_dir=out_path,
                        opacity=1,
                        with_labels=False)
    
if __name__ == '__main__':
    main()