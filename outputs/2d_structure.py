from rdkit import Chem
from rdkit.Chem import Draw

smiles = "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@@H]1[C@H]([C@H]([C@](O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4"  

molecule = Chem.MolFromSmiles(smiles)

Draw.MolToImage(molecule)

image = Draw.MolToImage(molecule)

# image.save("molecule_structure.pdf", "PDF")