"""
Setup and validation script for Multiple Sequence Alignment Pipeline.
Uses a reliable method to verify ClustalW functionality by testing actual alignment capabilities.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import tempfile
import textwrap

class AlignmentSetup:
    def __init__(self):
        """Initialize setup parameters and paths"""
        self.base_dir = Path("/dados/home/tesla-dados/multione")
        self.clustalw_path = self.base_dir / "clustalw-2.1/src/clustalw2"
        self.balibase_dir = self.base_dir / "BAliBASE/RV30"
        self.results_dir = self.base_dir / "results"
        
        # Define results directory structure
        self.results_subdirs = {
            "clustalw": self.results_dir / "clustalw",
            "muscle": self.results_dir / "muscle",
            "data": self.results_dir / "data",
            "logs": self.results_dir / "logs"
        }
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure detailed logging with both file and console output"""
        self.results_subdirs["logs"].mkdir(parents=True, exist_ok=True)
        log_file = self.results_subdirs["logs"] / "setup.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def create_test_sequences(self) -> Path:
        """
        Create a small test FASTA file for ClustalW validation.
        Returns path to temporary test file.
        """
        test_sequences = """>Seq1
ACGTACGTACGT
>Seq2
ACGTACGTAGCT
"""
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix='.fasta')
        with os.fdopen(fd, 'w') as temp:
            temp.write(test_sequences)
        return Path(path)

    def verify_clustalw(self) -> bool:
        """
        Verify ClustalW installation by performing a small test alignment.
        This is more reliable than checking help output or version flags.
        """
        try:
            # First check if binary exists and is executable
            if not self.clustalw_path.exists():
                logging.error(f"ClustalW binary not found at: {self.clustalw_path}")
                return False
                
            if not os.access(self.clustalw_path, os.X_OK):
                logging.error(f"ClustalW binary is not executable: {self.clustalw_path}")
                logging.info(f"Try running: chmod +x {self.clustalw_path}")
                return False

            # Create test input file
            test_file = self.create_test_sequences()
            output_file = test_file.with_suffix('.aln')
            
            try:
                # Run ClustalW with explicit input/output files
                cmd = [
                    str(self.clustalw_path),
                    "-INFILE=" + str(test_file),
                    "-OUTFILE=" + str(output_file),
                    "-QUIET"  # Suppress normal ClustalW output
                ]
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10  # Give it more time than before
                )
                
                # Check if alignment was created successfully
                if output_file.exists() and output_file.stat().st_size > 0:
                    logging.info("ClustalW test alignment completed successfully")
                    logging.info(f"Using ClustalW at: {self.clustalw_path}")
                    return True
                else:
                    logging.error("ClustalW failed to create alignment output")
                    if process.stderr:
                        logging.error(f"Error output: {process.stderr}")
                    return False
                    
            finally:
                # Clean up temporary files
                test_file.unlink(missing_ok=True)
                output_file.unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            logging.error("ClustalW process timed out while performing test alignment")
            return False
        except Exception as e:
            logging.error(f"Error during ClustalW verification: {str(e)}")
            return False

    def verify_balibase(self) -> bool:
        """
        Verify BAliBASE reference set availability and structure.
        """
        try:
            if not self.balibase_dir.exists():
                logging.error(f"BAliBASE directory not found: {self.balibase_dir}")
                return False
                
            # Check for .tfa files
            tfa_files = list(self.balibase_dir.glob("*.tfa"))
            if not tfa_files:
                logging.error("No .tfa files found in BAliBASE directory")
                return False
            
            # Provide detailed information about found files    
            logging.info(f"Found {len(tfa_files)} sequence files in BAliBASE directory")
            example_files = ', '.join(f.name for f in tfa_files[:3])
            logging.info(f"First few files: {example_files}...")
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying BAliBASE dataset: {str(e)}")
            return False

    def create_directory_structure(self) -> bool:
        """Create and verify all required directories for the pipeline."""
        try:
            for name, path in self.results_subdirs.items():
                path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created/verified directory: {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating directory structure: {str(e)}")
            return False

    def check_permissions(self) -> bool:
        """Verify read/write permissions for all required directories."""
        try:
            directories = [self.base_dir, self.balibase_dir, self.results_dir]
            directories.extend(self.results_subdirs.values())
            
            for directory in directories:
                if not os.access(directory, os.R_OK):
                    logging.error(f"Cannot read from directory: {directory}")
                    logging.info(f"Try: chmod +r {directory}")
                    return False
                if not os.access(directory, os.W_OK):
                    logging.error(f"Cannot write to directory: {directory}")
                    logging.info(f"Try: chmod +w {directory}")
                    return False
                    
            logging.info("Directory permissions verified successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error checking permissions: {str(e)}")
            return False

    def run_setup(self) -> bool:
        """Run complete setup and validation process with clear status updates."""
        logging.info("Starting alignment pipeline setup...")
        
        # Define our verification steps
        checks = [
            (self.verify_clustalw, "ClustalW verification"),
            (self.verify_balibase, "BAliBASE verification"),
            (self.create_directory_structure, "Directory structure creation"),
            (self.check_permissions, "Permission verification")
        ]
        
        # Run each check and track overall success
        success = True
        for check_func, description in checks:
            logging.info(f"Running {description}...")
            if not check_func():
                logging.error(f"{description} failed")
                success = False
                break

        if success:
            logging.info("Setup completed successfully")
        return success

def main():
    """Main setup function with detailed status reporting."""
    setup = AlignmentSetup()
    
    if setup.run_setup():
        print("\nSetup successful! Your environment is ready for the alignment pipeline.")
        print("\nVerified components:")
        print(f"✓ ClustalW: {setup.clustalw_path}")
        print(f"✓ BAliBASE: {setup.balibase_dir}")
        print(f"✓ Results:  {setup.results_dir}")
        print("\nYou can now proceed with running the alignment pipeline.")
        return 0
    else:
        print("\nSetup failed. Please check the detailed error messages above.")
        print(f"Full logs available at: {setup.results_dir}/logs/setup.log")
        return 1

if __name__ == "__main__":
    sys.exit(main())