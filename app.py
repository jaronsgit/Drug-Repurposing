import streamlit as st
import pandas as pd
from Bio import SeqIO, Phylo, AlignIO, Entrez
from Bio.Seq import Seq
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
import plotly.express as px
import matplotlib.pyplot as plt
import io
import re
import time
import numpy as np
import os
import pickle
from pathlib import Path
import torch as th
import torch.nn.functional as fn
import csv
from chembl import ChemblAPI
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import base64
import hashlib
import py3Dmol
from stmol import showmol
from openai import OpenAI
from functools import wraps
import html
import uuid
import json

# Initialize session state for chat and analysis caching
def init_session_state():
    """Initialize all session state variables"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'chat_visible' not in st.session_state:
        st.session_state.chat_visible = False
        
    if 'chat_client' not in st.session_state:
        try:
            key_file = open('samba_key.txt', 'r')
            st.session_state.chat_client = OpenAI(
                api_key=key_file.readline().rstrip(),
                base_url="https://api.sambanova.ai/v1",
            )
            key_file.close()
        except Exception as e:
            st.error(f"Failed to initialize chat client: {str(e)}")
    
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}

# Caching decorator
def prevent_rerun(func):
    """Decorator to prevent function from rerunning"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = f"cache_{func.__name__}_{str(args)}_{str(kwargs)}"
        if cache_key not in st.session_state.analysis_cache:
            st.session_state.analysis_cache[cache_key] = func(*args, **kwargs)
        return st.session_state.analysis_cache[cache_key]
    return wrapper

# Chat interface functions
def handle_chat_submit():
    """Handle chat form submission"""
    if st.session_state.chat_input and st.session_state.chat_input.strip():
        user_message = st.session_state.chat_input.strip()
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = st.session_state.chat_client.chat.completions.create(
                model="Meta-Llama-3.1-70B-Instruct",
                messages=[{"role": "user", "content": user_message}],
                stream=False
            )
            
            assistant_message = response.choices[0].message.content
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            
        except Exception as e:
            st.error(f"Failed to generate response: {str(e)}")
        
        st.session_state.chat_input = ""

def create_chat_interface():
    """Create a chat interface using Streamlit components"""
    st.markdown("""
        <style>
            .chat-message {
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e6f3ff;
                margin-left: 20%;
            }
            .assistant-message {
                background-color: #f0f0f0;
                margin-right: 20%;
            }
            .chat-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
            }
            .chat-toggle {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1001;
            }
        </style>
    """, unsafe_allow_html=True)

    # Chat interface container
    chat_container = st.sidebar.container()
    
    # Chat toggle in main interface
    if st.sidebar.button("💬 Toggle Chat", key="chat_toggle"):
        st.session_state.chat_visible = not st.session_state.chat_visible
        st.rerun()  # Changed from experimental_rerun() to rerun()
    
    if st.session_state.chat_visible:
        with chat_container:
            st.markdown("### SeqCure Assistant")
            
            # Display chat messages
            for msg in st.session_state.chat_messages:
                msg_class = "user-message" if msg["role"] == "user" else "assistant-message"
                st.markdown(
                    f"""<div class="chat-message {msg_class}">
                        {html.escape(msg["content"])}
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            # Chat input form
            with st.form(key="chat_form", clear_on_submit=True):
                st.text_input("Message", key="chat_input")
                submit_button = st.form_submit_button("Send", on_click=handle_chat_submit)

# Existing helper functions with @prevent_rerun decorator where appropriate
@prevent_rerun
def get_sequence_hash(sequence):
    sequence_str = str(sequence)
    return hashlib.md5(sequence_str.encode('utf-8')).hexdigest()

@prevent_rerun
def calculate_nucleotide_freq(sequence):
    total = len(sequence)
    frequencies = {
        'A': sequence.count('A') / total * 100,
        'T': sequence.count('T') / total * 100,
        'G': sequence.count('G') / total * 100,
        'C': sequence.count('C') / total * 100
    }
    return frequencies

@prevent_rerun
def gc_content(sequence):
    gc = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    return (gc / total) * 100

def validate_fasta(fasta_str):
    try:
        with io.StringIO(fasta_str) as handle:
            record = next(SeqIO.parse(handle, "fasta"))
            return True
    except:
        return False

@prevent_rerun
def perform_filtered_blast_search(sequence):
    cache_dir = Path("cache")
    sequence_hash = get_sequence_hash(sequence)
    cache_file = cache_dir / f"blast_results_{sequence_hash}.pkl"
    
    try:
        cache_dir.mkdir(exist_ok=True)
        
        if cache_file.exists():
            st.info("Loading BLAST results from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        st.info("No cached results found. Performing BLAST search...")
        entrez_query = "txid10239[Organism:exp] NOT txid2697049[Organism]"
        result_handle = NCBIWWW.qblast(
            "blastn", 
            "nt", 
            sequence,
            entrez_query=entrez_query,
            hitlist_size=10,
            expect=1e-50
        )
        
        result_text = result_handle.read()
        with open(cache_file, 'wb') as f:
            pickle.dump(result_text, f)
        
        return io.StringIO(result_text)
        
    except Exception as e:
        st.error(f"BLAST search failed: {str(e)}")
        return None


@prevent_rerun
def extract_sequences_from_blast(blast_results):
    """Extract sequences from BLAST results with metadata including species names"""
    if isinstance(blast_results, str):
        blast_results = io.StringIO(blast_results)
    
    blast_records = NCBIXML.parse(blast_results)
    sequences = []
    metadata = []
    seen_accessions = set()
    
    for record in blast_records:
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect > 1e-50:
                    continue
                    
                accession = alignment.title.split('|')[1]
                if accession not in seen_accessions:
                    seen_accessions.add(accession)
                    try:
                        Entrez.email = "your_email@example.com"  # Replace with your email
                        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
                        record = next(SeqIO.parse(handle, "genbank"))
                        handle.close()
                        
                        organism_name = record.annotations.get('organism', 'Unknown species')
                        alignment_score = hsp.score
                        
                        handle_fasta = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
                        seq_record = next(SeqIO.parse(handle_fasta, "fasta"))
                        handle_fasta.close()
                        
                        sequences.append(seq_record)
                        metadata.append({
                            'accession': accession,
                            'species': organism_name,
                            'e_value': hsp.expect,
                            'identity': (hsp.identities / hsp.align_length) * 100,
                            'alignment_score': alignment_score
                        })
                    except Exception as e:
                        st.warning(f"Could not fetch sequence {accession}: {str(e)}")
                        continue
    
    return sequences, metadata

@prevent_rerun
def cache_entrez_fetch(db, id_val, rettype, retmode, cache_dir="cache"):
    """Helper function to cache Entrez fetch results"""
    cache_dir = Path(cache_dir)
    params_str = f"{db}_{id_val}_{rettype}_{retmode}"
    params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"entrez_{params_hash}.pkl"
    
    try:
        cache_dir.mkdir(exist_ok=True)
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        handle = Entrez.efetch(db=db, id=id_val, rettype=rettype, retmode=retmode)
        result = handle.read()
        handle.close()
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
        
    except Exception as e:
        st.warning(f"Could not fetch/cache Entrez results for {id_val}: {str(e)}")
        return None

@prevent_rerun
def get_top_unique_species(metadata, n=5):
    """Get top n unique species based on alignment score"""
    df = pd.DataFrame(metadata)
    top_species = (df.sort_values('identity', ascending=False)
                    .drop_duplicates(subset=['species'])
                    .head(n))
    return top_species[['species', 'identity']]

@prevent_rerun
def create_phylogenetic_tree(query_sequence, similar_sequences):
    """Create phylogenetic tree using Biopython's built-in tools"""
    try:
        sequences = [query_sequence] + similar_sequences
        alignment = MultipleSeqAlignment([])
        min_length = min(len(s.seq) for s in sequences)
        
        for seq in sequences:
            seq_str = str(seq.seq)[:min_length]
            short_id = seq.id.split('.')[0]
            record = SeqRecord(
                Seq(seq_str),
                id=short_id[:15],
                name=short_id[:15],
                description=""
            )
            alignment.append(record)

        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)
        
        constructor = DistanceTreeConstructor(calculator)
        tree = constructor.build_tree(alignment)
        
        fig, ax = plt.subplots(figsize=(24, 18))
        Phylo.draw(tree, axes=ax, show_confidence=False, 
                  label_func=lambda x: x.name,  # Use the full name as label
                  do_show=False)  # Prevent automatic display
        
        # Increase font size for labels and adjust specific labels
        adjustment = 1  # Small adjustment value for label positions
        for text in ax.texts:
            
            if 'inner' in text.get_text().lower():
                text.set_visible(False)
            else:
                text.set_fontsize(35)  # Larger font for sequence labels
            
        # Remove tick labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Add more space between labels and tree
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.15)  # Add 15% more space on the right
        
        # Adjust layout to prevent overlapping
        plt.tight_layout(pad=2.0)  
        
        return fig, alignment
    
    
    except Exception as e:
        st.error(f"Error creating phylogenetic tree: {str(e)}")
        return None, None

@prevent_rerun
def extract_genes_from_gtf(gtf_file, status_placeholder, progress_bar):
    """Extract genes from GTF file with visual progress updates"""
    status_placeholder.write("Running gene finder...")
    progress_bar.progress(20)
    time.sleep(1)
    
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf_data = pd.read_csv(gtf_file, sep="\t", comment='#', header=None, names=columns)
    
    status_placeholder.write("Extracting gene information...")
    progress_bar.progress(50)
    time.sleep(1)
    
    genes_data = gtf_data[gtf_data['feature'] == 'gene']

    genes = []
    for attr in genes_data['attribute']:
        for field in attr.split(';'):
            if 'gene_name' in field:
                gene_name = field.split('"')[1]
                genes.append(gene_name)
                break
    
    progress_bar.progress(80)
    status_placeholder.write("Found closely related viral genes.")
    return genes

@prevent_rerun
def display_image(image_file, status_placeholder, progress_bar):
    """Display image with visual progress updates"""
    status_placeholder.write("Loading image...")
    progress_bar.progress(85)
    time.sleep(1)
    
    img = plt.imread(image_file)
    
    status_placeholder.write("Displaying image...")
    progress_bar.progress(90)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.axis('off')
    return fig

@st.cache_data
def fetch_compound_data(compounds):
    """Cached function to fetch compound data"""
    api = ChemblAPI()
    compound_data = {}
    
    for compound in compounds:
        try:
            metadata = api.get_compound_metadata(compound)
            if metadata and 'smile' in metadata:
                compound_data[compound] = metadata
        except Exception as e:
            st.warning(f"Could not fetch data for {compound}: {str(e)}")
    
    return compound_data

@st.cache_data
def get_cached_chemical_properties(smiles):
    """Cached function to calculate chemical properties"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {
            'Molecular Weight': round(Descriptors.ExactMolWt(mol), 2),
            'MolLogP': round(Descriptors.MolLogP(mol), 2),
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Topological Polar Surface Area': round(Descriptors.TPSA(mol), 2),
            'Number of Rings': Descriptors.RingCount(mol)
        }
        return properties
    except Exception as e:
        st.error(f"Error calculating properties: {str(e)}")
        return None

@st.cache_data
def get_cached_molecule_image(smiles):
    """Cached function to generate molecule image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error generating molecule image: {str(e)}")
        return None

@prevent_rerun
def drug_repurposing(container):
    """Complete drug repurposing function with error handling and persistent updates"""
    try:
        # Step 1: Initial Data Loading
        container.info("Step 1/8: Loading related viruses and viral targets...")
        COV_disease_list = [
            'Disease::SARS-CoV2 E',
            'Disease::SARS-CoV2 M',
            'Disease::SARS-CoV2 N',
            'Disease::SARS-CoV2 Spike',
            'Disease::SARS-CoV2 nsp1',
            'Disease::SARS-CoV2 nsp10',
            'Disease::SARS-CoV2 nsp11',
            'Disease::SARS-CoV2 nsp12',
            'Disease::SARS-CoV2 nsp13',
            'Disease::SARS-CoV2 nsp14',
            'Disease::SARS-CoV2 nsp15',
            'Disease::SARS-CoV2 nsp2',
            'Disease::SARS-CoV2 nsp4',
            'Disease::SARS-CoV2 nsp5',
            'Disease::SARS-CoV2 nsp5_C145A',
            'Disease::SARS-CoV2 nsp6',
            'Disease::SARS-CoV2 nsp7',
            'Disease::SARS-CoV2 nsp8',
            'Disease::SARS-CoV2 nsp9',
            'Disease::SARS-CoV2 orf10',
            'Disease::SARS-CoV2 orf3a',
            'Disease::SARS-CoV2 orf3b',
            'Disease::SARS-CoV2 orf6',
            'Disease::SARS-CoV2 orf7a',
            'Disease::SARS-CoV2 orf8',
            'Disease::SARS-CoV2 orf9b',
            'Disease::SARS-CoV2 orf9c',
            'Disease::MESH:D045169',
            'Disease::MESH:D045473',
            'Disease::MESH:D001351',
            'Disease::MESH:D065207',
            'Disease::MESH:D028941',
            'Disease::MESH:D058957',
            'Disease::MESH:D006517'
        ]
        time.sleep(2.3)
        # Step 2: Drug Database Loading
        container.info("Step 2/8: Loading drug database and treatment relationships...")
        drug_list = []
        try:
            with open("./infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug','ids'])
                for row_val in reader:
                    drug_list.append(row_val['drug'])
        except Exception as e:
            raise Exception(f"Error loading drug database: {str(e)}")

        treatment = ['Hetionet::CtD::Compound:Disease','GNBR::T::Compound:Disease']
        time.sleep(1.6)
        # Step 3: Entity Mapping
        container.info("Step 3/8: Loading and mapping drug-disease entity relationships...")
        entity_map = {}
        entity_id_map = {}
        relation_map = {}
        
        with open('./embed/entities.tsv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
            for row_val in reader:
                entity_map[row_val['name']] = int(row_val['id'])
                entity_id_map[int(row_val['id'])] = row_val['name']
                
        with open('./embed/relations.tsv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
            for row_val in reader:
                relation_map[row_val['name']] = int(row_val['id'])
        time.sleep(0.9)
        # Step 4: ID Processing        
        container.info("Step 4/8: Processing drug and disease identifiers...")
        drug_ids = []
        disease_ids = []
        for drug in drug_list:
            drug_ids.append(entity_map[drug])
            
        for disease in COV_disease_list:
            disease_ids.append(entity_map[disease])

        treatment_rid = [relation_map[treat] for treat in treatment]
        time.sleep(0.5)
        # Step 5: Loading Neural Network Embeddings
        container.info("Step 5/8: Generating drug-disease entity embeddings...")
        entity_emb = np.load('./embed/DRKG_TransE_l2_entity.npy')
        rel_emb = np.load('./embed/DRKG_TransE_l2_relation.npy')

        drug_ids = th.tensor(drug_ids).long()
        disease_ids = th.tensor(disease_ids).long()
        treatment_rid = th.tensor(treatment_rid)

        drug_emb = th.tensor(entity_emb[drug_ids])
        treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]
        time.sleep(5)
        # Step 6: Computing Drug-Disease Relationships
        container.info("Step 6/8: Computing drug-disease relationship scores...")
        gamma = 12.0
        def transE_l2(head, rel, tail):
            score = head + rel - tail
            return gamma - th.norm(score, p=2, dim=-1)

        scores_per_disease = []
        dids = []
        for rid in range(len(treatment_embs)):
            treatment_emb = treatment_embs[rid]
            for disease_id in disease_ids:
                disease_emb = entity_emb[disease_id]
                score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
                scores_per_disease.append(score)
                dids.append(drug_ids)
        time.sleep(1)
        # Step 7: Score Processing and Ranking
        container.info("Step 7/8: Ranking and prioritizing potential drug candidates...")
        scores = th.cat(scores_per_disease)
        dids = th.cat(dids)

        idx = th.flip(th.argsort(scores), dims=[0])
        scores = scores[idx].numpy()
        dids = dids[idx].numpy()
        
        _, unique_indices = np.unique(dids, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_dids = dids[topk_indices]
        proposed_scores = scores[topk_indices]
        time.sleep(1)
        # Step 8: Clinical Trial Analysis
        container.info("Step 8/8: Cross-referencing with clinical trial database...")
        clinical_drugs = []
        with open('./COVID19_clinical_trial_drugs.tsv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name','drug_id'])
            for row_val in reader:
                clinical_drugs.append({
                    'rank': None,
                    'name': row_val['drug_name'],
                    'score': None,
                    'drug_id': row_val['drug_id']
                })
        
        # Prepare results
        results = []
        clinical_results = []
        
        for i in range(topk):
            drug = int(proposed_dids[i])
            score = proposed_scores[i]
            drug_id = entity_id_map[drug][10:17]
            
            results.append({
                'drug': entity_id_map[drug],
                'score': -score
            })
            
            # Check if drug is in clinical trials
            for clinical_drug in clinical_drugs:
                if clinical_drug['drug_id'] == drug_id:
                    clinical_results.append({
                        'rank': i,
                        'name': clinical_drug['name'],
                        'score': -score
                    })

        container.success("✅ Drug repurposing analysis complete!")
        return pd.DataFrame(results), pd.DataFrame(clinical_results)
        
    except Exception as e:
        import traceback
        container.error(f"Detailed error in drug repurposing:\n{traceback.format_exc()}")
        raise

@prevent_rerun
def get_protein_structure_viewer(pdb_id):
    """Create a py3Dmol viewer for the protein structure"""
    try:
        pdb_file = f'{pdb_id}.pdb'
        if not os.path.exists(pdb_file):
            return None, f"Error: File {pdb_file} not found."
        
        viewer = py3Dmol.view(width=800, height=600)
        
        with open(pdb_file, 'r') as pdb:
            pdb_content = pdb.read()
        
        viewer.addModel(pdb_content, 'pdb')
        viewer.setStyle({'chain': ['A', 'C', 'D', 'P', 'T']}, 
                       {'cartoon': {'color': 'lightgrey', 'opacity': 0.5}})
        viewer.setStyle({'hetflag': True}, 
                       {'stick': {'colorscheme': 'orangeCarbon'}})
        
        viewer.zoomTo({'hetflag': True})
        viewer.zoom(0.5)
        
        html_str = f"""
        <div style="height: 600px; width: 100%;">
            <script src="https://3dmol.org/build/3Dmol-min.js"></script>
            <div id="viewport" style="height: 100%; width: 100%;"></div>
            <script>
                {viewer.js()}
            </script>
        </div>
        """
        
        return html_str, None
        
    except Exception as e:
        return None, f"Error rendering protein structure: {str(e)}"

def render_mol(pdb):
    """Create a py3Dmol viewer with custom styling"""
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb, "pdb")
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setBackgroundColor('white')
    view.zoomTo()
    return view


@prevent_rerun
def run_sequence_analysis(sequence_data):
    """Wrapper for sequence analysis section"""
    results = {}
    
    try:
        # Basic sequence analysis
        results['frequencies'] = calculate_nucleotide_freq(str(sequence_data.seq))
        results['gc_content'] = gc_content(str(sequence_data.seq))
        
        # BLAST analysis
        blast_results = perform_filtered_blast_search(str(sequence_data.seq))
        if blast_results:
            similar_sequences, metadata = extract_sequences_from_blast(blast_results)
            if similar_sequences:
                results['similar_sequences'] = similar_sequences
                results['metadata'] = metadata
                results['tree'], results['alignment'] = create_phylogenetic_tree(
                    sequence_data, 
                    similar_sequences
                )
    except Exception as e:
        st.error(f"Error in sequence analysis: {str(e)}")
        return None
        
    return results

@prevent_rerun
def run_gene_analysis(gtf_file='Sars_cov_2.ASM985889v3.101.gtf'):
    """Wrapper for gene analysis section"""
    try:
        genes = extract_genes_from_gtf(gtf_file)
        return {'genes': genes}
    except Exception as e:
        st.error(f"Error in gene analysis: {str(e)}")
        return None

@prevent_rerun
def run_drug_analysis():
    """Wrapper for drug repurposing analysis"""
    try:
        results, clinical_results = drug_repurposing(st)
        return {
            'results': results,
            'clinical_results': clinical_results
        }
    except Exception as e:
        st.error(f"Error in drug analysis: {str(e)}")
        return None
    
def chat_endpoint():
    try:
        data = json.loads(st.query_params.get('data', '{}'))
        messages = data.get('messages', [])
        model = data.get('model', "Meta-Llama-3.1-70B-Instruct")
        
        response = st.session_state.chat_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        
        return {"content": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}, 500

def main():
    # Initialize session state
    init_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="SeqCure",
        page_icon="🧬",
        layout="wide"
    )
    
    # Create main content container
    main_content = st.container()
    
    # Check if we're handling a chat endpoint
    if 'endpoint' in st.query_params and st.query_params['endpoint'] == 'chat':
        return chat_endpoint()
    
    with main_content:
        st.title("SeqCure")
        st.write("Upload or paste a FASTA file containing a viral genome sequence")
        
        # Add file upload and sequence input
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Choose a FASTA file", type=['fasta', 'fa'])
        with col2:
            pasted_sequence = st.text_area("Paste your FASTA sequence here", height=70)
        
        # Process input
    sequence_data = None
    fasta_analysis_complete = False
    
    if uploaded_file or pasted_sequence:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        try:
            # Step 1: Load and validate sequence
            status_placeholder.info("Step 1: Loading and validating sequence...")
            if uploaded_file:
                content = uploaded_file.read().decode()
                if validate_fasta(content):
                    sequence_data = next(SeqIO.parse(io.StringIO(content), "fasta"))
                else:
                    st.error("Invalid FASTA format in uploaded file")
                    
            elif pasted_sequence:
                if validate_fasta(pasted_sequence):
                    sequence_data = next(SeqIO.parse(io.StringIO(pasted_sequence), "fasta"))
                else:
                    st.error("Invalid FASTA format in pasted sequence")
            
            if sequence_data:
                # Step 2: Perform BLAST Search
                progress_bar.progress(33)
                status_placeholder.info("Step 2: Performing BLAST search for related viruses...")
                blast_results = perform_filtered_blast_search(str(sequence_data.seq))
                
                if blast_results:
                    progress_bar.progress(66)
                    status_placeholder.info("Step 3: Creating phylogenetic tree...")
                    similar_sequences, metadata = extract_sequences_from_blast(blast_results)
                    
                    if similar_sequences:
                        tree_fig, alignment = create_phylogenetic_tree(sequence_data, similar_sequences)
                        
                        progress_bar.progress(100)
                        status_placeholder.success("Analysis complete!")
                        
                        # Clear progress indicators
                        time.sleep(1)
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        
                        # Display results
                        st.success(f"Found {len(similar_sequences)} closely related viral sequences")
                        
                        # Create results tabs
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Sequence Analysis",
                            "Similar Sequences",
                            "Phylogenetic Tree",
                            "Raw Data"
                        ])

                        with tab1:
                            st.subheader("Sequence Analysis")
                            frequencies = calculate_nucleotide_freq(str(sequence_data.seq))
                            
                            freq_df = pd.DataFrame({
                                'Nucleotide': list(frequencies.keys()),
                                'Frequency (%)': list(frequencies.values())
                            })
                            
                            fig = px.bar(freq_df, x='Nucleotide', y='Frequency (%)',
                                        title='Nucleotide Distribution',
                                        color='Nucleotide')
                            st.plotly_chart(fig)
                                                        
                        with tab2:
                            st.subheader("Similar Viral Sequences")
                            df = pd.DataFrame(metadata)
                            df['Identity %'] = df['identity'].round(2)
                            df['E-value'] = df['e_value'].apply(lambda x: f"{x:.2e}")
                            display_df = df[['species', 'Identity %', 'E-value']].copy()
                            display_df.columns = ['Species', 'Identity %', 'E-value']
                            st.dataframe(display_df)
                            
                            # Add section for known human coronaviruses
                            st.subheader("Related Viral Genomes")
                            corona_species = [
                                {"species": "Human coronavirus 229E (hCoV-229E)", "description": "Common cold coronavirus"},
                                {"species": "Human coronavirus NL63 (hCoV-NL63)", "description": "Upper respiratory tract infection"},
                                {"species": "Human coronavirus OC43 (hCoV-OC43)", "description": "Common cold coronavirus"},
                                {"species": "Human coronavirus HKU1 (hCoV-HKU1)", "description": "Upper respiratory infection"},
                                {"species": "SARS coronavirus (SARS-CoV)", "description": "Severe Acute Respiratory Syndrome"},
                                {"species": "Middle East respiratory syndrome coronavirus (MERS-CoV)", "description": "Middle East Respiratory Syndrome"}
                            ]
                            
                            for idx, corona in enumerate(corona_species, 1):
                                st.markdown(
                                    f"""
                                    **{idx}. {corona['species']}**  
                                    {corona['description']}
                                    """
                                )
                        
                        with tab3:
                            st.subheader("Phylogenetic Tree of Related Viruses")
                            if tree_fig:
                                st.pyplot(tree_fig)
                            else:
                                st.error("Could not generate phylogenetic tree")
                        
                        with tab4:
                            st.subheader("Raw Sequence Data")
                            st.write(f"Sequence ID: {sequence_data.id}")
                            st.write(f"Sequence Length: {len(sequence_data.seq)} bp")
                            if st.checkbox("Show raw sequence"):
                                st.text_area("Raw Sequence", str(sequence_data.seq), height=200)
                        
                        fasta_analysis_complete = True
                        
                    else:
                        st.error("No similar sequences found")
                else:
                    st.error("BLAST search failed")
        
        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error(f"An error occurred during processing: {str(e)}")

    else:
        st.info("Please upload or paste a FASTA sequence to begin analysis")

    # Gene Analysis Section - only show if FASTA analysis is complete
    if fasta_analysis_complete:
        st.markdown("---")
        st.header("Gene Annotation Analysis")
        st.write("View gene annotations and organization")
        
        # Hard-coded file paths
        gtf_file = 'Sars_cov_2.ASM985889v3.101.gtf'
        image_file = 'SARS-CoV-2_Genes.png'
        
        # Create placeholders for status updates and progress bar
        gene_progress_placeholder = st.empty()
        gene_status_placeholder = st.empty()
        gene_progress_bar = gene_progress_placeholder.progress(0)
        
        try:
            # Extract genes
            gene_names = extract_genes_from_gtf(gtf_file, gene_status_placeholder, gene_progress_bar)
            
            # Create tabs for different views
            gene_tab, viz_tab = st.tabs(["Gene List", "Gene Visualization"])
            
            with gene_tab:                
                # Create a DataFrame from the gene names
                gene_df = pd.DataFrame(gene_names, columns=["Gene Names"])
                
                # Display the DataFrame as a table
                st.dataframe(gene_df)

            with viz_tab:
                # Display the image
                fig = display_image(image_file, gene_status_placeholder, gene_progress_bar)
                st.pyplot(fig)
            
            # Complete the progress bar
            gene_progress_bar.progress(100)
            gene_status_placeholder.success("Gene analysis complete!")
            
            # Clear progress indicators after completion
            time.sleep(1)
            gene_progress_placeholder.empty()
            gene_status_placeholder.empty()
            
        except Exception as e:
            gene_progress_placeholder.empty()
            gene_status_placeholder.empty()
            st.error(f"An error occurred during gene analysis: {str(e)}")

        # Drug Repurposing Analysis Section
        st.markdown("---")
        st.header("Drug Repurposing Analysis")
        st.write("Analyzing potential drug candidates for treatment.")
        
        # Check for required files before starting
        required_files = [
            Path("./infer_drug.tsv"),
            Path("./embed/entities.tsv"),
            Path("./embed/relations.tsv"),
            Path("./embed/DRKG_TransE_l2_entity.npy"),
            Path("./embed/DRKG_TransE_l2_relation.npy"),
            Path("./COVID19_clinical_trial_drugs.tsv")
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            st.error("Missing required files:")
            for f in missing_files:
                st.error(f"- {f}")
            return
            
        # Create a container for persistent step updates
        with st.container():
            try:
                # Run drug repurposing analysis
                _, clinical_results_df = drug_repurposing(st)
                
                st.header('Top Repurposing Candidates')
                
                try:
                    # Initialize session state for selected compound if not exists
                    if 'selected_compound' not in st.session_state:
                        st.session_state.selected_compound = None
                    
                    compounds = clinical_results_df["name"].to_list()
                    # Fetch compound data using cached function
                    compound_data = fetch_compound_data(compounds)
                    
                    # Create DataFrame only from successfully fetched data
                    results = [data for data in compound_data.values() if data]
                    if not results:
                        st.error("No compound data could be retrieved")
                        return
                        
                    data_df = pd.DataFrame.from_records(results)
                    data_df = clinical_results_df.merge(data_df, left_on="name", right_on="input_name", how="left")
                    
                    # Display the main table first
                    display_df = data_df.drop(columns=["name", "input_name", "smile", "rank"]).rename(columns={
                        "compound_name": "Compound", 
                        "score": "Score",
                        "targets": "Targets",
                        "disease_names": "Diseases",
                        "max_phase": "Trial Phase",
                    })
                    st.dataframe(display_df.set_index("Compound"))
                    
                    # Create compound selection using radio buttons instead of regular buttons
                    st.write("Select a compound to view chemical properties:")
                    selected = st.radio(
                        "Choose compound",
                        options=compounds,
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    
                    # Update session state when selection changes
                    if selected and selected in compound_data:
                        st.session_state.selected_compound = selected
                        st.session_state.compound_smile = compound_data[selected]['smile']
                    
                    # Display chemical properties section if a compound is selected
                    if st.session_state.selected_compound:
                        st.markdown("---")
                        st.header(f"Chemical Properties: {st.session_state.selected_compound}")
                        
                        col1, col2 = st.columns([0.3, 0.7])
                        
                        with col1:
                            # Display chemical properties using cached function
                            properties = get_cached_chemical_properties(st.session_state.compound_smile)
                            if properties:
                                for prop, value in properties.items():
                                    st.metric(prop, value)
                            else:
                                st.error("Could not calculate chemical properties")
                        
                        with col2:
                            # Display 2D structure using cached function
                            img_str = get_cached_molecule_image(st.session_state.compound_smile)
                            if img_str:
                                st.image(f"data:image/png;base64,{img_str}", 
                                       caption="2D Structure")
                            else:
                                st.error("Could not generate molecular structure")
                        
                        # Display SMILES
                        with st.expander("Show SMILES notation"):
                            st.code(st.session_state.compound_smile)

                        # Add AlphaFold section after chemical properties
                        st.markdown("---")
                        st.header("AlphaFold2 Drug-Protein Interaction")
                        
                        # Load the PDB file
                        try:
                            with open('./7l1f.pdb', 'r') as f:
                                pdb_data = f.read()

                            st.write("Protein-Drug complex visualization:")

                            # Initialize the 3D viewer
                            view = py3Dmol.view(width=800, height=500)
                            view.addModel(pdb_data, 'pdb')
                            
                            # Default styles for chains and heteroatoms
                            view.setStyle({'chain': ['A', 'C', 'D', 'P', 'T']}, 
                                        {'cartoon': {'color': 'grey', 'opacity': 0.75}})
                            view.setStyle({'hetflag': True}, {'stick': {'colorscheme': 'orangeCarbon'}})
                            
                            # Zoom settings
                            view.zoomTo({'hetflag': True})
                            view.zoom(0.5)

                            # Display the model
                            showmol(view, height=500, width=800)

                            # Add interaction options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.checkbox("Show binding site"):
                                    # Highlight the binding site
                                    view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'})

                            with col2:
                                style = st.selectbox(
                                    "Visualization style",
                                    ["cartoon", "stick", "sphere", "line"]
                                )
                                # Update visualization style based on user selection
                                if style:
                                    view.setStyle({}, {style: {}})
                                    
                            # Re-render after making updates
                            view.render()
                                    
                        except FileNotFoundError:
                            st.error("PDB structure file not found")
                        except Exception as e:
                            st.error(f"Error loading molecular structure: {str(e)}")

                        # Literature Search Section
                        st.markdown("---")
                        st.header("Literature Search")

                        try:
                            # Read the CSV file
                            df_literature = pd.read_csv('biorxiv_results.csv')
                            
                            # Display the table with clickable links
                            st.write("Recent bioRxiv papers related to drug mechanisms and interactions:")
                            st.dataframe(
                                df_literature.set_index("Title"),
                                column_config={
                                    "Link": st.column_config.LinkColumn(display_text="Open Paper")
                                }
                            )
                            
                            # Optional: Add filtering capability with interactive dataframe for better UX
                            with st.expander("🔍 Filter papers"):
                                search_term = st.text_input(
                                    "Filter by keyword",
                                    placeholder="Enter keyword...",
                                    key="paper_search",
                                )
                                if search_term:
                                    filtered_df = df_literature[
                                        df_literature['Title'].str.contains(
                                            search_term, 
                                            case=False, 
                                            na=False
                                        )
                                    ]
                                    st.dataframe(
                                        filtered_df.set_index("Title"),
                                        column_config={
                                            "Link": st.column_config.LinkColumn(display_text="Open Paper")
                                        }
                                    )
                            
                        except FileNotFoundError:
                            st.error("Literature data file not found")
                        except Exception as e:
                            st.error(f"Error loading literature data: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")

                        # Patent Search Section
                        st.markdown("---")
                        st.header("Patent Search")

                        try:
                            # Read the CSV file
                            df_patents = pd.read_csv('outputs/first_10_patents.csv')
                            
                            # Clean up titles (remove ellipsis and truncation)
                            df_patents['Title'] = df_patents['Title'].apply(
                                lambda x: x.replace('…', '').strip()
                            )
                            
                            # Display the table with clickable links
                            st.write("Recent patents related to drug development and applications:")
                            st.dataframe(
                                df_patents.set_index("Title"),
                                column_config={
                                    "Link": st.column_config.LinkColumn(display_text="Open Patent")
                                }
                            )
                            
                            # Add information about the data
                            st.caption("Source: Patent database search results")

                            # Optional: Add filtering capability
                            with st.expander("🔍 Filter patents"):
                                search_term = st.text_input(
                                    "Filter by keyword",
                                    placeholder="Enter keyword...",
                                    key="patent_search",
                                )
                                if search_term:
                                    filtered_df = df_patents[
                                        df_patents['Title'].str.contains(
                                            search_term, 
                                            case=False, 
                                            na=False
                                        )
                                    ]
                                    st.dataframe(
                                        filtered_df.set_index("Title"),
                                        column_config={
                                            "Link": st.column_config.LinkColumn(display_text="Open Patent")
                                        }
                                    )
                            
                        except FileNotFoundError:
                            st.error("Patent data file not found")
                        except Exception as e:
                            st.error(f"Error loading patent data: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")                        
                
                except Exception as e:
                    st.error(f"Error in drug information section: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

            except Exception as e:
                import traceback
                st.error("An error occurred during drug analysis. Details:")
                st.code(traceback.format_exc(), language="python")

            
    if not (uploaded_file or pasted_sequence):
        # Footer
        st.markdown("---")
        st.markdown("""
            ### About this tool
            This tool provides comprehensive analysis of COVID-19 genomic data:
            - Sequence analysis and phylogenetic relationships
            - Identification of closely related viral sequences
            - Gene annotation analysis and visualization
            - Drug repurposing analysis
            - Detailed sequence statistics
        """)
    # Add chat interface
    create_chat_interface()

if __name__ == "__main__":
    main()