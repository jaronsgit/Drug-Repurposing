import streamlit as st
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import plotly.express as px
import io
import re

def calculate_nucleotide_freq(sequence):
    """Calculate nucleotide frequencies in the sequence"""
    total = len(sequence)
    frequencies = {
        'A': sequence.count('A') / total * 100,
        'T': sequence.count('T') / total * 100,
        'G': sequence.count('G') / total * 100,
        'C': sequence.count('C') / total * 100
    }
    return frequencies

def gc_content(sequence):
    """Calculate GC content percentage"""
    gc = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    return (gc / total) * 100

def validate_fasta(fasta_str):
    """Validate if the input is in FASTA format"""
    try:
        with io.StringIO(fasta_str) as handle:
            record = next(SeqIO.parse(handle, "fasta"))
            return True
    except:
        return False

# Set page config
st.set_page_config(
    page_title="COVID-19 Genome Analyzer",
    layout="wide"
)

# Main title
st.title("COVID-19 Genome Sequence Analyzer")
st.write("Upload or paste a FASTA file containing COVID-19 genome sequence")

# Create two columns for input methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload FASTA File")
    uploaded_file = st.file_uploader("Choose a FASTA file", type=['fasta', 'fa'])
    
with col2:
    st.subheader("Or Paste FASTA Sequence")
    pasted_sequence = st.text_area("Paste your FASTA sequence here", height=200)

# Process the input
sequence_data = None
if uploaded_file:
    try:
        content = uploaded_file.read().decode()
        if validate_fasta(content):
            sequence_data = next(SeqIO.parse(io.StringIO(content), "fasta"))
        else:
            st.error("Invalid FASTA format in uploaded file")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

elif pasted_sequence:
    try:
        if validate_fasta(pasted_sequence):
            sequence_data = next(SeqIO.parse(io.StringIO(pasted_sequence), "fasta"))
        else:
            st.error("Invalid FASTA format in pasted sequence")
    except Exception as e:
        st.error(f"Error processing pasted sequence: {str(e)}")

# Analysis section
if sequence_data:
    st.success("Sequence loaded successfully!")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Basic Info", "Composition Analysis", "Sequence Details"])
    
    with tab1:
        st.subheader("Sequence Information")
        st.write(f"Sequence ID: {sequence_data.id}")
        st.write(f"Sequence Length: {len(sequence_data.seq)} bp")
        st.write(f"GC Content: {gc_content(str(sequence_data.seq)):.2f}%")
    
    with tab2:
        st.subheader("Nucleotide Composition")
        frequencies = calculate_nucleotide_freq(str(sequence_data.seq))
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Nucleotide': list(frequencies.keys()),
            'Frequency (%)': list(frequencies.values())
        })
        
        # Create an interactive bar plot using plotly
        fig = px.bar(df, x='Nucleotide', y='Frequency (%)',
                    title='Nucleotide Distribution',
                    color='Nucleotide')
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Sequence Details")
        if st.checkbox("Show raw sequence"):
            st.text_area("Raw Sequence", str(sequence_data.seq), height=200)
        
        # Add sequence statistics
        st.write("### Sequence Statistics")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.metric("Total Length", f"{len(sequence_data.seq)} bp")
            st.metric("GC Content", f"{gc_content(str(sequence_data.seq)):.2f}%")
        
        with stats_col2:
            st.metric("AT Content", 
                     f"{100 - gc_content(str(sequence_data.seq)):.2f}%")
            st.metric("Non-standard bases", 
                     len(re.findall(r'[^ATGC]', str(sequence_data.seq))))

else:
    st.info("Please upload or paste a FASTA sequence to begin analysis")

# Add footer with information
st.markdown("---")
st.markdown("""
    ### About this tool
    This tool analyzes COVID-19 genome sequences in FASTA format. It provides:
    - Basic sequence information
    - Nucleotide composition analysis
    - GC content calculation
    - Detailed sequence statistics
""")