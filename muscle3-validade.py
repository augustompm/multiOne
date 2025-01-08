class AlignmentPipeline:
    def __init__(self, 
                 balibase_dir: str = "/dados/home/tesla-dados/multione/BAliBASE/RV30",
                 results_dir: str = "/dados/home/tesla-dados/multione/results",
                 clustalw_bin: str = "/dados/home/tesla-dados/multione/clustalw-2.1/src/clustalw2",
                 muscle_bin: str = "/dados/home/tesla-dados/multione/muscle3.8.31/src/muscle"):
        """
        Initialize alignment pipeline with paths and parameters
        
        Parameters
        ----------
        balibase_dir : str
            Directory containing BAliBASE reference alignments
        results_dir : str 
            Directory to store results
        clustalw_bin : str
            Path to ClustalW executable
        muscle_bin : str
            Path to MUSCLE executable
        """
        self.balibase_dir = Path(balibase_dir)
        self.results_dir = Path(results_dir)
        self.clustalw_bin = Path(clustalw_bin)
        self.muscle_bin = Path(muscle_bin)
        
        # Create results directories
        self.clustalw_dir = self.results_dir / "clustalw"
        self.muscle_dir = self.results_dir / "muscle" 
        self.data_dir = self.results_dir / "data"
        
        # Verify tool installations
        self._verify_tools()
        
        for directory in [self.clustalw_dir, self.muscle_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize scoring matrix for evaluations
        self.similarity_matrix = self._create_iub_matrix()

    def _verify_tools(self):
        """
        Verify that both ClustalW and MUSCLE are properly installed.
        Provides helpful installation instructions if tools are missing.
        """
        # Check ClustalW
        if not self.clustalw_bin.exists():
            msg = f"""
            ClustalW not found at {self.clustalw_bin}
            Please install ClustalW:
            1. Download from http://www.clustal.org/clustal2/
            2. Extract and build:
               tar xzf clustalw-2.1.tar.gz
               cd clustalw-2.1
               ./configure
               make
            """
            raise FileNotFoundError(msg)

        # Check MUSCLE
        if not self.muscle_bin.exists():
            msg = f"""
            MUSCLE not found at {self.muscle_bin}
            Please install MUSCLE:
            1. Download MUSCLE:
               wget https://drive5.com/muscle/downloads3.8.31/muscle3.8.31_i86linux64.tar.gz
            2. Extract and make executable:
               tar xzf muscle3.8.31_i86linux64.tar.gz
               chmod +x muscle3.8.31_i86linux64
               mv muscle3.8.31_i86linux64 /dados/home/tesla-dados/multione/muscle3.8.31/src/muscle
            """
            raise FileNotFoundError(msg)

    def run_muscle(self, input_file: Path) -> Path:
        """
        Run MUSCLE alignment with parameters optimized for BAliBASE Reference 3.
        
        Our parameter choices focus on handling divergent sequences effectively:
        
        Iteration Control:
        - maxiters=3: Multiple refinement iterations for better accuracy
        - diags=True: Uses diagonal optimization for divergent sequences
        
        Divergent Sequence Handling:
        - sv=True: Stabilizes alignment of divergent regions
        - weight1=clustalw: Uses ClustalW-style sequence weighting
        - sueff=0.4: Lower value for divergent sequences
        """
        output_file = self.muscle_dir / f"{input_file.stem}.aln"
        
        try:
            muscle_cline = MuscleCommandline(
                cmd=str(self.muscle_bin),  # Use correct path to MUSCLE executable
                input=str(input_file),
                out=str(output_file),
                maxiters=3,
                diags=True,
                sv=True,
                weight1="clustalw",
                cluster1="upgmb",
                sueff=0.4,
                root1="pseudo",
                maxtrees=2,
                clw=True
            )
            logging.info(f"Running MUSCLE on {input_file} with optimized parameters")
            subprocess.run(str(muscle_cline).split(), check=True)
            return output_file
            
        except subprocess.CalledProcessError as e:
            logging.error(f"MUSCLE alignment failed for {input_file}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error running MUSCLE: {str(e)}")
            raise