###For running the code
- provide necessary env
  - running on RTX 3090, 24GB; CUDA 11.3.

  - install necessary packages, e.g., torch, etc.
    
    (1) conda create -n ConGCL python==3.8
  
    (2) conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
        
        https://pytorch.org/get-started/previous-versions
  
    (3) conda install pyg -c pyg

        https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

    (4) install other packages in requirements.txt

        pip install -r requirements.txt

  - config some custom paths.
  
    (1) for simple_param/sp.py, modify the `local_dir` according to your local env; 
        e.g., `<path_to_code>/ConGCL/param`.

    (2) for train.py, please remember provide the `ppr_base_path` in your arguments 
        in the purpose of storing the ppr scores. e.g., `<path_to_code>/ConGCL/pGRACE/subgraph`.
    Note: <path_to_code> denotes the path of ConGCL dir.

- running the code over different datasets.

  (1) for Cora
    
  ```xml
    python train.py --device cuda:0 --dataset Cora --param local:cora.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```
    
  (2) for Amazon-Computers

  ```xml
    python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon_computers.json --drop_scheme degree --mu 1 --gamma 0.85 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```
    
  (3) for Amazon-Photo

  ```xml
    python train.py --device cuda:0 --dataset Amazon-Photo --param local:amazon_photo.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```
    
  (4) for WikiCS

  ```xml
    python train.py --device cuda:0 --dataset WikiCS --param local:wikics.json --drop_scheme degree --mu 6 --gamma 0.8 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```
    
  (5) for Coauthor-CS

  ```xml
    python train.py --device cuda:0 --dataset Coauthor-CS --param local:coauthor_cs.json --drop_scheme degree --mu 4 --gamma 0.95 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```

  (6) for PubMed

  ```xml
    python train.py --device cuda:0 --dataset PubMed --param local:pubmed.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```

  (7) for Coauthor-Phy

  ```xml
    python train.py --device cuda:0 --dataset Coauthor-Phy --param local:coauthor_phy.json --drop_scheme degree --mu 4 --gamma 0.95 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph
  ```
  
- (Ablation Study) running the code over different datasets.

  (1) for Cora
    
  ```xml
    python train.py --device cuda:0 --dataset Cora --param local:cora.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```
  
  ```xml
    python train.py --device cuda:0 --dataset Cora --param local:cora.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```
    
  (2) for Amazon-Computers

  ```xml
    python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon_computers.json --drop_scheme degree --mu 1 --gamma 0.85 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```
  
  ```xml
    python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon_computers.json --drop_scheme degree --mu 1 --gamma 0.85 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```
    
  (3) for Amazon-Photo

  ```xml
    python train.py --device cuda:0 --dataset Amazon-Photo --param local:amazon_photo.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```

  ```xml
    python train.py --device cuda:0 --dataset Amazon-Photo --param local:amazon_photo.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```
    
  (4) for WikiCS

  ```xml
    python train.py --device cuda:0 --dataset WikiCS --param local:wikics.json --drop_scheme degree --mu 6 --gamma 0.8 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```
  
  ```xml
    python train.py --device cuda:0 --dataset WikiCS --param local:wikics.json --drop_scheme degree --mu 6 --gamma 0.8 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```
    
  (5) for Coauthor-CS

  ```xml
    python train.py --device cuda:0 --dataset Coauthor-CS --param local:coauthor_cs.json --drop_scheme degree --mu 4 --gamma 0.95 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```
  
  ```xml
    python train.py --device cuda:0 --dataset Coauthor-CS --param local:coauthor_cs.json --drop_scheme degree --mu 4 --gamma 0.95 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```

  (6) for PubMed

  ```xml
    python train.py --device cuda:0 --dataset PubMed --param local:pubmed.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```

  ```xml
    python train.py --device cuda:0 --dataset PubMed --param local:pubmed.json --drop_scheme degree --mu 4 --gamma 0.9 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```

  (7) for Coauthor-Phy

  ```xml
    python train.py --device cuda:0 --dataset Coauthor-Phy --param local:coauthor_phy.json --drop_scheme degree --mu 4 --gamma 0.95 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_2sim 1
  ```

  ```xml
    python train.py --device cuda:0 --dataset Coauthor-Phy --param local:coauthor_phy.json --drop_scheme degree --mu 4 --gamma 0.95 --ppr_base_path <path_to_code>/ConGCL/pGRACE/subgraph --rm_alpha 1
  ```

- Notes
  - The code will download benchmark datasets to `~/datasets`,
    it may fails to download due to network issue.
  - Feel free to put out any issue that you meet with while running the code.
    we will attempt to provide solutions. Thanks.