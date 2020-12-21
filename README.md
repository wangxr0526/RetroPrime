# RetroPrime
This is the code for the "RetroPrime: A Chemistry-Inspired and Transformer-based Method for Retro-synthesis Predictions" \
To implement our models we were based on OpenNMT-py (v0.4.1).
# Install requirements
Create a new conda environment:
```
conda install create -n retroprime_env python=3.6
conda activate retroprime_env
conda install -c conda-forge rdkit
conda install pandas tqdm six
conda install pytorch==1.5.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
Then,
```
pip install -e .
cd retroprime/dataprocess/packeage/SmilesEnumerator/
pip install -e .
```
This step installs the Smiles enumerator. https://github.com/EBjerrum/SMILES-enumeration, or you can also use RDKit's own enumeration function to replace these parts of this code.

Then,
```
cd retroprime/transformer_model/
pip install -e .
```
# Data Processing
```
cd retroprime/dataprocess/
```
You can follow the sequence number of the *.sh script. Or you can change the file path in the script to handle your own reaction data. Reaction data like this:
```
[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6].CCOCC>C(N(CC)CC)C>[CH2:5]([S:7]([O:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8])[CH3:6]
```

