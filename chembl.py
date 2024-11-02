from chembl_webresource_client.new_client import new_client

SYNONYMS = {
    "Middle East respiratory syndrome-related coronavirus": "MERS",
    "Severe acute respiratory syndrome-related coronavirus": "SARS-CoV2",
    "Zaire ebolavirus (strain Mayinga-76) (ZEBOV) (Zaire Ebola virus)": "Ebola",
}

class ChemblAPI:
    def __init__(self) -> None:
        self.client = new_client

    def get_compound_metadata(self, compound_name: str) -> dict:
        results = self.client.molecule.search(compound_name)
        molecule_entries = [result for result in results if result["therapeutic_flag"]]
        chembl_ids = [entry["molecule_chembl_id"] for entry in molecule_entries]

        # Fetch drug mechanisms related to the molecule
        target_names = set()
        disease_names = set()
        for chembl_id in chembl_ids:
            mechanisms_entries = self.client.mechanism.filter(molecule_chembl_id=chembl_id)
            for entry in mechanisms_entries:
                target_names.add(entry["mechanism_of_action"])
            indication_entries = self.client.drug_indication.filter(molecule_chembl_id=chembl_id, max_phase_for_ind__gt=3)
            for entry in indication_entries:
                disease_names.add(entry["mesh_heading"])
            
        return {
            "input_name": compound_name,
            "compound_name": molecule_entries[0]["pref_name"],
            "targets": sorted(target_names),
            "disease_names": sorted(disease_names),
            "max_phase": "IV",
            "smile": molecule_entries[0]["molecule_structures"]["canonical_smiles"]
        }


if __name__ == "__main__":
    print(ChemblAPI().get_compound_metadata("Methylprednisolone"))
