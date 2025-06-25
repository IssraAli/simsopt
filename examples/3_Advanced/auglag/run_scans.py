import subprocess
import random
import os
import argparse
import sys

def run_parameter_scan(num_runs):
    """
    Runs the auglag_qa_scan.py script N times with random parameters.

    Args:
        num_runs (int): The number of times to run the optimization script.
    """
    base_output_dir = "scan_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Starting parameter scan for {num_runs} runs...")
    random.seed(1) 
    for i in range(num_runs):
        # Generate random values within the specified ranges
        cs_threshold = random.uniform(0.02, 0.5)
        length_target = random.uniform(20, 35)

        # Create a unique output directory for this run
        run_dir_name = f"run_{i}_cs_{cs_threshold:.4f}_len_{length_target:.2f}"
        run_output_dir = os.path.join(base_output_dir, run_dir_name)
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Log file for the subprocess output
        log_file_path = os.path.join(run_output_dir, 'output.log')

        print("-" * 70)
        print(f"Starting Run {i+1}/{125+num_runs}")
        print(f"  Coil-Surface Threshold: {cs_threshold:.4f}")
        print(f"  Coil Length Target: {length_target:.2f}")
        print(f"  Output Directory: {run_output_dir}")
        print(f"  Log File: {log_file_path}")
        
        command = [
            sys.executable,  # Use the same python interpreter that runs this script
            "auglag_qa_scan_call.py",
            "--cs_threshold", str(cs_threshold),
            "--length_target", str(length_target),
            "--out_dir", run_output_dir
        ]

        try:
            # Open the log file to write stdout and stderr
            with open(log_file_path, 'w') as log_file:
                # Run the subprocess
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                process.wait() # Wait for the subprocess to complete

            if process.returncode == 0:
                print(f"Run {i+1} completed successfully.")
            else:
                print(f"Run {i+1} failed with return code {process.returncode}. Check log for details.")

        except FileNotFoundError:
            print("\nError: 'auglag_qa_scan.py' not found.")
            print("Please ensure both 'run_scans.py' and 'auglag_qa_scan.py' are in the same directory.")
            return
        except Exception as e:
            print(f"An error occurred during run {i+1}: {e}")

    print("-" * 70)
    print("All runs completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple instances of a coil optimization script with random parameters.")
    parser.add_argument(
        "-N",
        "--num_runs",
        type=int,
        required=True,
        help="The number of optimization runs to perform."
    )
    args = parser.parse_args()
    
    run_parameter_scan(args.num_runs)
